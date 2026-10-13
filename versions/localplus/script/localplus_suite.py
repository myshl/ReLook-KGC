#!/usr/bin/env python3
"""Bounded, validation-only tuning; one held-out test after settings are frozen.

Old experiments are read-only. The local CSProm backbone is transferred exactly
into ONE ReLook model. This is continued training, not an ensemble or a claim of
equal training cost. Compare the refined backbone as well as the final model.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_suite import stage_options, write_json, memory_available

PYTHON = ROOT / '.venv/bin/python'
RELEASE = ROOT.parent / 'ReLook-KGC-unified-20261012'
STAGES = ['shared', 'hierarchical', 'visual', 'relation', 'query', 'prior', 'joint']


def environment():
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false',
               LD_LIBRARY_PATH=str(ROOT/'.cuda_lib') + ':' + env.get('LD_LIBRARY_PATH', ''))
    return env


def run(command, directory, gpu):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'status.json').exists():
        raise RuntimeError(f'Refusing to overwrite prior attempt: {directory}')
    write_json(directory/'command.json', command)
    while memory_available(gpu) < 32000:
        write_json(directory/'waiting.json', dict(status='waiting_for_32GB_free', gpu=gpu))
        print(f'Waiting for memory on GPU {gpu}: {directory}', flush=True)
        time.sleep(30)
    started = time.time()
    with (directory/'run.log').open('w') as f:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=f, stderr=subprocess.STDOUT)
        write_json(directory/'status.json', dict(status='running', pid=child.pid, gpu=gpu,
                                                started=started))
        print(f'START gpu={gpu} pid={child.pid} {directory}', flush=True)
        code = child.wait()
    write_json(directory/'status.json', dict(status='finished' if code == 0 else 'failed',
                                           gpu=gpu, returncode=code, seconds=time.time()-started))
    if code:
        raise RuntimeError(f'Failed: {directory}/run.log; no silent OOM retries')


def options(dataset, stage, smoke):
    backbone = stage in ('ce_refine', 'lar_refine')
    cfg = stage_options(dataset, 'all' if backbone else stage)
    # Keep original architecture, smoothing, data, seed and effective batch.
    # Only the declared optimization recipe changes.
    cfg.update(batch_size=256, accumulate_grad_batches=2, val_batch_size=128,
               gpu_memory_fraction=.30, keep_initial_checkpoint=True,
               relook_truth_set_weight=0., min_epochs=2,
               early_stop_patience=3, early_stop_delta=.00005)
    if backbone:
        cfg.update(lr=5e-5, epoch=12, relook_base_weight=1)
        if stage == 'lar_refine':
            cfg.update(n_lar=8, alpha=.05)
    elif stage in ('shared', 'hierarchical'):
        cfg.update(lr=2e-5, epoch=8, relook_adapter_norm_weight=1e-3)
    elif stage == 'visual':
        cfg.update(epoch=8, lr=1e-4)
    else:
        cfg.update(epoch=4, min_epochs=1, lr=5e-5, early_stop_patience=2)
    if smoke:
        cfg.update(epoch=1, min_epochs=1, early_stop_patience=0,
                   limit_train_batches=2, limit_val_batches=1, limit_test_batches=1)
    return cfg


def train_stage(dataset, stage, output, gpu, previous, smoke):
    directory = output/stage
    cfg = options(dataset, stage, smoke)
    cfg.update(gpu=gpu, save_dir=str(directory/'checkpoints'),
               metrics_output=str(directory/'valid.json'))
    if stage in ('ce_refine', 'lar_refine'):
        cfg['csprom_init_path'] = previous
    else:
        cfg['continue_path'] = previous
    command = [str(PYTHON), str(ROOT/'main.py')]
    for key, value in cfg.items():
        command += [f'-{key}'] + ([] if value is True else [str(value)])
    run(command, directory, gpu)
    result = json.loads((directory/'valid.json').read_text())
    assert result['split'] == 'valid'
    write_json(directory/'complete.json', result)
    print(f'DONE {dataset}/{stage}: {result["exact_performance"]["mean ranking"]}', flush=True)
    return result


def preflight(checkpoint, output, gpu):
    import torch
    from final_analysis import load_runtime, prepare_batch
    from models.relook_model import ReLookKG
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30, gpu)
    config, reference, dm, gt = load_runtime(checkpoint, gpu)
    cfg = copy.deepcopy(config)
    cfg.model_family = 'relook'
    cfg.relook_role_adapters = False
    cfg.relook_backbone_only = True
    texts = dict(ent_names=reference.ent_names, rel_names=reference.rel_names,
                 ent_descs=reference.ent_descs)
    torch.manual_seed(cfg.seed)
    converted = ReLookKG(cfg, texts, gt)
    converted.load_csprom_initializer(checkpoint)
    converted.eval().to(f'cuda:{gpu}')
    tests = []
    with torch.inference_mode():
        for role, loader in enumerate(dm.val_dataloader()):
            batch = prepare_batch(next(iter(loader)), next(converted.parameters()).device)
            batch = [x[:16] for x in batch]
            before = reference(*batch)[0]
            after = converted._shared_backbone_forward(*batch)[0]
            error = float((before-after).abs().max())
            assert error < 1e-6, (role, error)
            tests.append(dict(role='head' if role else 'tail', queries=len(batch[0]),
                              max_abs_logit_error=error))
    write_json(output/'preflight.json', dict(status='passed', source_checkpoint=checkpoint,
                                           forward_equivalence=tests))
    print(json.dumps(tests), flush=True)


def final_evaluation(selected, output, gpu, smoke):
    import gc
    import numpy as np
    import torch
    from final_analysis import load_runtime, collect, calibrate, direction_metrics, probabilities
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30, gpu)
    out = output/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    limit = 1 if smoke else -1
    # Final checkpoint and visual policy are frozen using VALIDATION only.
    for stage in ('visual', 'prior', 'joint'):
        config, model, dm, gt = load_runtime(selected[stage], gpu)
        rows = collect(model, dm, gt, 'valid', sweep=True, limit=limit)
        chosen, grid = calibrate(rows)
        chosen.update(stage=stage, checkpoint=selected[stage])
        candidates.append(chosen)
        np.savez_compressed(out/f'{stage}_valid_cache.npz', **rows)
        write_json(out/f'{stage}_validation_grid.json', grid)
        del model, dm, rows
        gc.collect()
        torch.cuda.empty_cache()
    champion = max(candidates, key=lambda x: (x['mrr'], -x['invocation'], -x['blend']))
    write_json(out/'validation_candidates.json', candidates)
    write_json(out/'deployment_frozen_before_test.json', champion)
    config, model, dm, gt = load_runtime(champion['checkpoint'], gpu)
    config.relook_max_blend = champion['blend']
    config.relook_gate_threshold = champion['threshold']
    config.relook_tail_evidence_gain_cap = champion['tail_cap']
    config.relook_head_evidence_gain_cap = champion['head_cap']
    config.selection_policy = 'threshold'
    # Validate actual inference against the cached selection BEFORE held-out use.
    validation = collect(model, dm, gt, 'valid', limit=limit)
    cache = np.load(out/f'{champion["stage"]}_valid_cache.npz')
    mask = ((probabilities(cache, champion['tail_cap'], champion['head_cap']) >= champion['threshold'])
            & cache['eligible'])
    expected = np.where(mask, cache[f'visual_rank_{champion["blend"]}'], cache['base_rank'])
    assert np.array_equal(validation['rank'], expected), 'Validation cache/inference mismatch'
    write_json(out/'inference_replay_check.json', dict(exact_match=True, count=len(expected)))
    test = collect(model, dm, gt, 'test', limit=limit)
    np.savez_compressed(out/'full_filtered_test.npz', **test)
    measured = direction_metrics(test['rank'], test['role'])
    # The refined backbone is a required control, not a tuning candidate on test.
    del model, dm
    gc.collect()
    torch.cuda.empty_cache()
    _, backbone, dm, gt = load_runtime(selected['backbone'], gpu)
    control = collect(backbone, dm, gt, 'test', limit=limit)
    np.savez_compressed(out/'refined_backbone_filtered_test.npz', **control)
    import csv
    local = next(r for r in csv.DictReader((RELEASE/'data/main_results_locked.csv').open())
                 if r['dataset'] == config.dataset and r['stage'] == 'csprom')
    names = ('mrr', 'hit1', 'hit3', 'hit10')
    result = dict(status='smoke_only' if smoke else 'complete', dataset=config.dataset,
                  seed=config.seed, final=measured,
                  refined_backbone=direction_metrics(control['rank'], control['role']),
                  local={key: float(local[key]) for key in names}, deployment=champion,
                  delta_vs_local_points={key: 100*(measured['mean'][key]-float(local[key])) for key in names})
    result['exceeds_local_mrr'] = measured['mean']['mrr'] > float(local['mrr']) if not smoke else None
    result['exceeds_local_all_four'] = all(measured['mean'][key] > float(local[key]) for key in names) if not smoke else None
    write_json(out/'results.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['FB15k-237', 'YAGO15K'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--launch', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--phase', choices=['pipeline', 'preflight', 'final'], default='pipeline')
    args = p.parse_args()
    output = (args.output or ROOT/'experiments'/('smoke' if args.smoke else 'localplus')/args.dataset).resolve()
    manifest = json.loads((RELEASE/'release_manifest.json').read_text())
    checkpoint = manifest['datasets'][args.dataset]['checkpoints']['csprom']['path']
    if args.phase == 'preflight':
        preflight(checkpoint, output, args.gpu)
        return
    if args.phase == 'final':
        selected = json.loads((output/'selected_checkpoints.json').read_text())
        final_evaluation(selected, output, args.gpu, args.smoke)
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.launch:
        if (output/'launcher.json').exists():
            raise RuntimeError('Already launched; inspect PID/log instead of duplicating')
        command = [str(PYTHON), str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != '--launch']
        with (output/'queue.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        info = dict(pid=child.pid, command=command, output=str(output))
        write_json(output/'launcher.json', info)
        print(json.dumps(info, indent=2))
        return
    protocol = dict(version='localplus-20261012', dataset=args.dataset, seed=41504,
                    initialization='exact local CSProm single-backbone checkpoint transfer; no ensemble',
                    source_checkpoint=manifest['datasets'][args.dataset]['checkpoints']['csprom'],
                    selection='full validation MRR; incoming checkpoint eligible at every stage; no test selection',
                    held_out='all entities, both directions, all-known filtered, original argsort tie rule',
                    budget='two 12-epoch refinement recipes, early stop; 8-epoch experts/visual, 4-epoch routers',
                    controls='local published run and refined backbone; extra tuning budget is explicitly not matched',
                    preserved='DB15K frozen full model; all historical artifacts and manuscript unchanged',
                    options={s: options(args.dataset, s, args.smoke) for s in ['ce_refine', 'lar_refine']+STAGES})
    write_json(output/'protocol.json', protocol)
    lock = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for subdir in ('models', 'script', 'tests', 'reference')
            for path in (ROOT/subdir).rglob('*.py')}
    lock.update({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob('*.py')})
    write_json(output/'source_lock.json', lock)
    subprocess.check_call([str(PYTHON), '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, env=environment())
    command = [str(PYTHON), str(Path(__file__).resolve()), '--phase', 'preflight',
               '--dataset', args.dataset, '--gpu', str(args.gpu), '--output', str(output)]
    run(command, output/'preflight_run', args.gpu)
    if args.preflight_only:
        return
    trials = {}
    for stage in (['lar_refine'] if args.smoke else ['ce_refine', 'lar_refine']):
        trials[stage] = train_stage(args.dataset, stage, output, args.gpu, checkpoint, args.smoke)
    winner = max(trials, key=lambda s: trials[s]['exact_performance']['mean ranking']['mrr'])
    selected = dict(backbone=trials[winner]['checkpoint'])
    write_json(output/'backbone_selection.json', dict(winner=winner, trials=trials, split='valid'))
    previous = selected['backbone']
    for stage in STAGES:
        result = train_stage(args.dataset, stage, output, args.gpu, previous, args.smoke)
        selected[stage] = previous = result['checkpoint']
        write_json(output/'selected_checkpoints.json', selected)
    command = [str(PYTHON), str(Path(__file__).resolve()), '--phase', 'final',
               '--dataset', args.dataset, '--gpu', str(args.gpu), '--output', str(output)]
    if args.smoke:
        command.append('--smoke')
    run(command, output/'final_evaluation', args.gpu)
    write_json(output/'complete.json', dict(status='smoke_only' if args.smoke else 'complete',
                                          results=str(output/'analysis/results.json')))


if __name__ == '__main__':
    main()
