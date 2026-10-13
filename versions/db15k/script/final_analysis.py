#!/usr/bin/env python3
"""Validation-only calibration, held-out selection controls, real CUDA timings."""
import argparse
import csv
from collections import defaultdict
import gc
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import BertTokenizer, logging as transformers_logging
transformers_logging.set_verbosity_error()
from helper import get_gt, read, read_file, read_name
from kgc_data import KGCDataModule
from models.relook_model import ReLookKG
from models.csprom_control import CSPromControl
from paper_suite import write_json

BLENDS = [.1, .25, .5, 1.]
THRESHOLDS = [0., .1, .25, .5, .75, 1.01]
CAPS = [-1., 0., .1, .25]


def load_runtime(checkpoint, gpu):
    payload = torch.load(checkpoint, map_location='cpu')
    config = payload['hyper_parameters']['configs']
    config.model_path = str(checkpoint)
    config.badcase_report = ''
    config.num_workers = 4
    config.val_batch_size = 128
    train, valid, test = [read(config, config.dataset_path, config.dataset, split+'2id.txt') for split in ['train', 'valid', 'test']]
    train_tail, train_head = get_gt(config, train)
    all_tail, all_head = get_gt(config, train+valid+test)
    gt = dict(train_tail_gt=train_tail, train_head_gt=train_head, all_tail_gt=all_tail, all_head_gt=all_head)
    names, rels = read_name(config, config.dataset_path, config.dataset)
    texts = dict(ent_names=names, rel_names=rels, ent_descs=read_file(config, config.dataset_path, config.dataset, 'entityid2description.txt', 'desc'))
    model = (CSPromControl if config.model_family == 'csprom' else ReLookKG)(config, texts, gt)
    model.load_state_dict(payload['state_dict'], strict=True)
    del payload
    model.eval().to(f'cuda:{gpu}')
    tok = BertTokenizer.from_pretrained(config.pretrained_model)
    dm = KGCDataModule(config, train, valid, test, texts, tok, gt)
    return config, model, dm, gt


def metrics(ranks):
    r = np.asarray(ranks, dtype=float)
    return dict(count=len(r), mrr=float(np.mean(1/r)), hit1=float(np.mean(r==1)),
                hit3=float(np.mean(r<=3)), hit10=float(np.mean(r<=10)), mr=float(r.mean()))


def direction_metrics(ranks, roles):
    return {name: metrics(ranks[mask]) for name,mask in [
        ('mean', np.ones(len(ranks), dtype=bool)), ('tail', roles==0), ('head', roles==1)]}


def prepare_batch(batch, device):
    return [batch[k].to(device, non_blocking=True) for k in ['ent_rel', 'source_ids', 'source_mask']]


def get_ranks(logits, labels, mask):
    # Same argsort tie convention as the training-time validation code.
    order = torch.argsort(logits.masked_fill(mask, -1e9), dim=1, descending=True)
    return (order == labels[:, None]).long().argmax(dim=1).add(1).cpu().numpy()


@torch.inference_mode()
def collect(model, dm, gt, split, sweep=False, limit=-1):
    config = model.configs
    device = next(model.parameters()).device
    loaders = dm.val_dataloader() if split == 'valid' else dm.test_dataloader()
    rows = defaultdict(list)
    old_policy = config.selection_policy
    if sweep:
        config.selection_policy = 'none'
    for role, loader in enumerate(loaders):
        for bi, batch in enumerate(loader):
            if limit > 0 and bi >= limit:
                break
            ent_rel, ids, attention = prepare_batch(batch, device)
            labels = torch.as_tensor(batch['tgt_ent'], device=device)
            output = model(ent_rel, ids, attention)
            official = isinstance(output, tuple)
            logits = output[0] if official else output['logits']
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite held-out logits')
            mask = torch.zeros_like(logits, dtype=torch.bool)
            lookup = gt['all_head_gt' if role else 'all_tail_gt']
            for i, (src, rel) in enumerate(ent_rel.tolist()):
                mask[i, lookup[(src, rel)]] = True
                mask[i, labels[i]] = False
            rows['rank'].append(get_ranks(logits, labels, mask))
            rows['role'].append(np.full(len(labels), role))
            rows['source'].append(ent_rel[:, 0].cpu().numpy())
            rows['relation'].append(ent_rel[:, 1].cpu().numpy())
            rows['target'].append(labels.cpu().numpy())
            rows['batch'].append(np.full(len(labels), role*100000+bi))
            if not official:
                rows['base_rank'].append(get_ranks(output['base_logits'], labels, mask))
                rows['shared_rank'].append(get_ranks(output['shared_logits'], labels, mask))
                rows['adapter_rank'].append(get_ranks(output['adapter_logits'], labels, mask))
                rows['invoked'].append(output['inspection_decision'].cpu().numpy())
            if sweep:
                base = output['base_logits']
                raw_visual = model.visual_verifier.score_all(model._visual_entities(ent_rel[:,0]),
                    model.visual_rel_embed(ent_rel[:,1]), model._visual_entities())
                for blend in BLENDS:
                    rows[f'visual_rank_{blend}'].append(get_ranks(base + blend*raw_visual, labels, mask))
                rows['eligible'].append(model.has_image_mask[ent_rel[:,0]].cpu().numpy())
                rows['prior'].append(output['visual_prior_probability'].cpu().numpy())
                rows['gain'].append(output['evidence_router_gain_logit'].cpu().numpy())
                values = base.float().topk(min(20, base.shape[1]), dim=1).values
                logp = values.log_softmax(dim=1)
                rows['entropy'].append((-(logp.exp()*logp).sum(dim=1)).cpu().numpy())
                rows['margin'].append((values[:,1]-values[:,0]).cpu().numpy())
    config.selection_policy = old_policy
    result = {k: np.concatenate(v) for k,v in rows.items()}
    if limit < 0:
        expected = 2*len(dm.valid if split=='valid' else dm.test)
        assert len(result['rank']) == expected, (len(result['rank']), expected)
    return result


def probabilities(rows, tail_cap, head_cap):
    p = np.clip(rows['prior'].astype(np.float32), 1e-5, 1-1e-5)
    gain = rows['gain'].copy()
    for role,cap in [(0, tail_cap), (1, head_cap)]:
        if cap >= 0:
            gain[rows['role']==role] = np.minimum(gain[rows['role']==role], cap)
    return 1/(1+np.exp(-np.clip(np.log(p/(1-p))+gain, -80, 80)))


def calibrate(rows):
    results = []
    for blend, threshold, tail, head in itertools.product(BLENDS, THRESHOLDS, CAPS, CAPS):
        selected = (probabilities(rows, tail, head) >= threshold) & rows['eligible']
        ranks = np.where(selected, rows[f'visual_rank_{blend}'], rows['base_rank'])
        results.append(dict(blend=blend, threshold=threshold, tail_cap=tail, head_cap=head,
                            invocation=float(selected.mean()), **metrics(ranks)))
    # Predeclared tie-break: accuracy, then fewer invocations, then smaller blend.
    best = max(results, key=lambda x: (x['mrr'], -x['invocation'], -x['blend']))
    return best, results


def policy_mask(rows, policy, budget, probability, seed, nrel):
    if policy == 'none':
        return np.zeros(len(probability), dtype=bool)
    if policy == 'always':
        return rows['eligible'].copy()
    if policy == 'random':
        key = (rows['source'].astype(np.int64)*(2*nrel)+rows['relation']+seed) & 0x7fffffff
        key = ((key ^ (key >> 16))*0x45d9f3b) & 0x7fffffff
        key = ((key ^ (key >> 16))*0x45d9f3b) & 0x7fffffff
        score = key ^ (key >> 16)
    else:
        score = probability if policy == 'learned' else rows[policy]
    result = np.zeros(len(probability), dtype=bool)
    for batch in np.unique(rows['batch']):
        inds = np.where(rows['batch']==batch)[0]
        eligible = inds[rows['eligible'][inds]]
        count = min(len(eligible), round(len(inds)*budget))
        result[eligible[np.argsort(-score[eligible], kind='stable')[:count]]] = True
    return result


def selection_results(rows, chosen, config):
    p = probabilities(rows, chosen['tail_cap'], chosen['head_cap'])
    rows_out = []
    policies = [('none', 0), ('always', 1), ('deployment', chosen['invocation'])]
    policies += list(itertools.product(['random','entropy','margin','learned'], [.25,.5,.75]))
    for policy,budget in policies:
        selected = (p >= chosen['threshold']) & rows['eligible'] if policy == 'deployment' else policy_mask(rows, policy, budget, p, config.seed, config.n_rel)
        ranks = np.where(selected, rows[f'visual_rank_{chosen["blend"]}'], rows['base_rank'])
        rr_gain = 1/ranks-1/rows['base_rank']
        rows_out.append(dict(policy=policy, budget=budget, actual_invocation=float(selected.mean()),
            mrr_gain_points=float(rr_gain.mean()*100), improved_fraction=float((rr_gain>0).mean()),
            harmed_fraction=float((rr_gain<0).mean()), mean_positive_rr_gain=float(np.maximum(rr_gain,0).mean()),
            mean_negative_rr_loss=float(np.maximum(-rr_gain,0).mean()),
            harmed_given_invoked=float(((rr_gain<0)&selected).sum()/max(selected.sum(),1)),
            metrics=direction_metrics(ranks, rows['role'])))
    return rows_out


@torch.inference_mode()
def benchmark(model, dm, chosen, gpu, smoke=False):
    # No unrelated process may share the timing GPU. This also avoids claiming
    # shared-GPU background load as a reproducible inference result.
    gpu_uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    procs = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    foreign = [line for line in procs.splitlines() if line.split(',')[0].strip()==gpu_uuid and int(line.split(',')[1]) != os.getpid()]
    if foreign and not smoke:
        return dict(status='blocked_shared_gpu', competing_processes=foreign)
    config = model.configs
    device = next(model.parameters()).device
    batches = []
    for loader in dm.test_dataloader():
        for i,batch in enumerate(loader):
            batches.append(prepare_batch(batch, device))
            if i == 1:
                break
    config.relook_max_blend = chosen['blend']
    config.relook_gate_threshold = chosen['threshold']
    config.relook_tail_evidence_gain_cap = chosen['tail_cap']
    config.relook_head_evidence_gain_cap = chosen['head_cap']
    policies = [('none',0),('always',1),('threshold',0)] + list(itertools.product(['random','entropy','margin','learned'],[.25,.5,.75]))
    if config.model_family == 'csprom' or not config.relook_role_adapters:
        policies = [('none', 0)]
    result = []
    for policy, budget in policies:
        config.selection_policy, config.selection_budget = policy, budget
        active_parameters = {}
        def capture(module, inputs):
            for parameter in module.parameters(recurse=False):
                active_parameters[id(parameter)] = parameter.numel()
        hooks = [module.register_forward_pre_hook(capture) for module in model.modules()]
        model(*batches[0])
        for hook in hooks:
            hook.remove()
        for i in range(1 if smoke else 4):
            model(*batches[i % len(batches)])
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings, invocations = [], []
        for repeat in range(1 if smoke else 5):
            for batch in batches:
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = model(*batch)
                torch.cuda.synchronize(device)
                timings.append((time.perf_counter()-start)/len(batch[0]))
                invocations.append(float(output['inspection_decision'].float().mean()) if isinstance(output,dict) else 0.)
        mean = float(np.mean(timings))
        result.append(dict(policy=policy,budget=budget,ms_per_query=mean*1000,
            std_ms_per_query=float(np.std(timings)*1000),p95_ms_per_query=float(np.percentile(timings,95)*1000),
            queries_per_second=1/mean,actual_invocation=float(np.mean(invocations)),
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
            executed_module_parameters=sum(active_parameters.values()),
            measurements=len(timings)))
    return dict(status='smoke_only_shared_gpu' if foreign else 'complete', competing_processes=foreign,
        device=torch.cuda.get_device_name(device),gpu_uuid=gpu_uuid,
        pytorch=torch.__version__,cuda=torch.version.cuda,precision='fp32',batch_size=128,
        scope='model-only; pretokenized resident GPU inputs; no filtering/ranking/data loading',
        parameter_count_definition='unique registered parameters of modules executed on the first measured batch; not FLOPs',
        instantiated_parameters=sum(p.numel() for p in model.parameters()),
        final_stage_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        rows=result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = args.suite/args.dataset/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads((args.suite/args.dataset/'selected_checkpoints.json').read_text())
    limit = 1 if args.smoke else -1
    main_results = {}
    efficiency_status = 'pending'
    for stage,checkpoint in selected.items():
        config, model, dm, gt = load_runtime(checkpoint, args.gpu)
        if stage != 'joint':
            rows = collect(model, dm, gt, 'test', limit=limit)
            main_results[stage] = direction_metrics(rows['rank'], rows['role'])
            np.savez_compressed(out/f'{stage}_test_ranks.npz', **rows)
            if stage in ['all','csprom']:
                baseline_setting=dict(blend=0, threshold=1.01, tail_cap=-1, head_cap=-1)
                write_json(out/f'{stage}_efficiency.json',benchmark(model,dm,baseline_setting,args.gpu,args.smoke))
        else:
            valid = collect(model, dm, gt, 'valid', sweep=True, limit=limit)
            chosen,grid = calibrate(valid)
            write_json(out/'validation_calibration_grid.json', grid)
            write_json(out/'deployment_config.json', chosen)
            np.savez_compressed(out/'joint_valid_cache.npz', **valid)
            test = collect(model, dm, gt, 'test', sweep=True, limit=limit)
            np.savez_compressed(out/'joint_test_cache.npz', **test)
            policies = selection_results(test, chosen, config)
            write_json(out/'selection_strategies.json', policies)
            deployment = next(row for row in policies if row['policy']=='deployment')
            main_results['full_validation_selected'] = deployment['metrics']
            fixed_mask = (probabilities(test,-1,-1)>=.25)&test['eligible']
            fixed_ranks = np.where(fixed_mask,test['visual_rank_0.5'],test['base_rank'])
            main_results['full_fixed_uncalibrated'] = direction_metrics(fixed_ranks,test['role'])
            main_results['full_router_no_visual'] = direction_metrics(test['base_rank'],test['role'])
            # Check held-out replay against the real forward on two full batches.
            config.relook_max_blend=chosen['blend']
            config.relook_gate_threshold=chosen['threshold']
            config.relook_tail_evidence_gain_cap=chosen['tail_cap']
            config.relook_head_evidence_gain_cap=chosen['head_cap']
            config.selection_policy='threshold'
            verify = collect(model, dm, gt, 'test', limit=1)
            first = (test['batch']==0)|(test['batch']==100000)
            p_test = probabilities(test,chosen['tail_cap'],chosen['head_cap'])
            mask = (p_test>=chosen['threshold'])&test['eligible']
            replay = np.where(mask,test[f'visual_rank_{chosen["blend"]}'],test['base_rank'])[first]
            if not np.array_equal(verify['rank'],replay):
                raise RuntimeError('Replay differs from actual selected forward; do not publish.')
            write_json(out/'replay_verification.json',dict(queries=len(replay),exact_match=True))
            efficiency = benchmark(model,dm,chosen,args.gpu,args.smoke)
            efficiency_status = efficiency['status']
            write_json(out/'actual_efficiency.json',efficiency)
        write_json(out/'main_results.json', main_results)
        print(f'Analysis finished: {args.dataset}/{stage}', flush=True)
        del model,dm,gt
        gc.collect()
        torch.cuda.empty_cache()
    with (out/'main_results.csv').open('w', newline='') as f:
        fields=['stage','direction','count','mrr','hit1','hit3','hit10','mr']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for stage, result in main_results.items():
            for direction, scores in result.items():
                writer.writerow(dict(stage=stage,direction=direction,**scores))
    write_json(out/'complete.json',dict(status='complete' if efficiency_status=='complete' else 'accuracy_complete_efficiency_pending',
        efficiency_status=efficiency_status,smoke=args.smoke,seed=41504))


if __name__ == '__main__':
    main()
