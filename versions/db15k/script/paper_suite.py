#!/usr/bin/env python3
"""Locked single-seed experiment DAG. All subprocesses use this project's venv."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
DATASETS = ['DB15K', 'YAGO15K', 'FB15k-237']
CHAIN = ['all', 'shared', 'hierarchical', 'visual', 'relation', 'query', 'prior', 'joint']
CONTROLS = ['mixed', 'early6', 'late6', 'csprom', 'role', 'shared_extended']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def stage_options(dataset, stage, smoke=False):
    backbone = stage in ['all', 'mixed', 'early6', 'late6', 'csprom']
    options = dict(
        dataset=dataset, dataset_path=str(ROOT / 'data/processed'),
        feature_dir=str(ROOT / 'features'), pretrained_model=str(ROOT / 'model_path/bert-base-uncased'),
        seed=41504, batch_size=512, val_batch_size=128, num_workers=4,
        desc_max_length=40, text_len=72, prompt_length=10, embed_dim=256,
        prompt_hidden_dim=128, k_w=16, k_h=16, alpha=0, n_lar=0,
        label_smoothing=.1, lr=5e-4 if backbone else 1e-4,
        epoch=(60 if dataset == 'FB15k-237' else 40) if backbone else (6 if stage in ['shared', 'role', 'hierarchical', 'shared_extended', 'visual'] else 3),
        check_val_every_n_epoch=1, min_epochs=10 if backbone else 1,
        early_stop_patience=6 if backbone else (3 if stage in ['shared', 'role', 'hierarchical', 'shared_extended', 'visual'] else 2),
        early_stop_delta=.0001, eval_split='valid', disable_progress_bar=True,
        relook_scope='global', relook_topk=128, relook_stat_topk=20,
        relook_backbone_prompt_mode='mixed' if stage == 'mixed' else 'progressive',
        relook_backbone_prompt_schedule=stage if stage in ['early6', 'late6'] else 'all',
        relook_truth_set_weight=1, relook_head_loss_weight=1, relook_train_direction='both',
        relook_gate_mode='legacy', relook_gate_threshold=1.01, relook_max_blend=0,
        relook_base_weight=0, relook_rerank_weight=0, relook_visual_weight=0,
        relook_budget_weight=0, relook_gate_weight=0, relook_role_adapter_weight=0,
        relook_role_router_weight=0, relook_adapter_norm_weight=0,
        relook_role_router_mode='full', relook_visual_utility_target='filtered',
    )
    if backbone:
        options.update(relook_backbone_only=True, relook_base_weight=1, use_fp16=True)
        if stage == 'csprom':
            options.update(model_family='csprom', n_lar=8, alpha=.1, relook_truth_set_weight=0)
    else:
        options.update(relook_role_adapters=True, relook_deep_experts=True,
                       relook_deep_expert_sharing='hierarchical', relook_deep_expert_layers=6,
                       relook_deep_expert_dim=64)
        if stage in ['shared', 'hierarchical', 'role', 'shared_extended']:
            sharing = {'shared_extended': 'shared'}.get(stage, stage)
            options.update(relook_adapter_only=True, relook_deep_expert_sharing=sharing,
                           relook_role_adapter_weight=1, relook_adapter_norm_weight=1e-4, use_fp16=True)
        elif stage == 'visual':
            options.update(relook_visual_only=True, relook_gate_mode='utility',
                           relook_gate_threshold=0, relook_min_gate=.5,
                           relook_max_blend=.5, relook_rerank_weight=1, relook_visual_weight=.2)
        elif stage == 'relation':
            options.update(relook_router_only=True, relook_role_router_mode='relation',
                           relook_role_router_target='rank', relook_role_router_weight=1)
        else:
            options.update(relook_role_router_mode='hybrid', relook_monotonic_query_router=True,
                           relook_query_residual_scale=.25)
            if stage == 'query':
                options.update(relook_router_only=True, relook_query_router_only=True,
                               relook_role_router_target='rank', relook_role_router_weight=1)
            else:
                options.update(relook_gate_mode='utility', relook_gate_threshold=.25,
                               relook_max_blend=.5, relook_min_gate=.5, relook_gate_weight=1)
                if stage == 'prior':
                    options.update(relook_gate_only=True)
                elif stage == 'joint':
                    options.update(relook_router_only=True, relook_joint_evidence_router=True)
                else:
                    raise ValueError(stage)
    if smoke:
        options.update(epoch=1, min_epochs=1, early_stop_patience=0,
                       limit_train_batches=4, limit_val_batches=1, limit_test_batches=1)
    return options


def memory_available(gpu):
    result = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    return int(result.strip())


def run_command(command, directory, gpu):
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=str(ROOT / '.cuda_lib') + ':' + env.get('LD_LIBRARY_PATH', ''),
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    write_json(directory / 'command.json', command)
    while memory_available(gpu) < 28000:
        print(f'Waiting for 28 GB free on GPU {gpu}: {directory.name}', flush=True)
        time.sleep(30)
    start = time.time()
    with (directory / 'run.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(directory / 'status.json', dict(status='running', pid=child.pid, gpu=gpu, start=start))
        print(f'START gpu={gpu} pid={child.pid} {directory}', flush=True)
        code = child.wait()
    write_json(directory / 'status.json', dict(status='process_finished' if code == 0 else 'failed',
                returncode=code, gpu=gpu, seconds=time.time()-start))
    if code:
        raise RuntimeError(f'{directory}: exit {code}; inspect run.log. No silent retries or protocol changes.')


def run_stage(dataset, stage, gpu, output, previous=None, smoke=False):
    directory = output / dataset / stage
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'complete.json').read_text())['checkpoint']
    if (directory / 'status.json').exists():
        raise RuntimeError(f'Incomplete prior run: {directory}. Audit it; do not overwrite or silently restart.')
    options = stage_options(dataset, stage, smoke)
    options.update(gpu=gpu, save_dir=str(directory / 'checkpoints'), metrics_output=str(directory / 'valid.json'))
    if previous:
        options['continue_path'] = previous
    command = [str(PYTHON), str(ROOT / 'main.py')]
    for name, value in options.items():
        command += [f'-{name}'] + ([] if value is True else [str(value)])
    run_command(command, directory, gpu)
    result = json.loads((directory / 'valid.json').read_text())
    convergence = directory / 'checkpoints/convergence.jsonl'
    if not convergence.exists():
        raise RuntimeError(f'Missing convergence audit: {directory}')
    rows = [json.loads(line) for line in convergence.read_text().splitlines()]
    if len(rows) < options['min_epochs']:
        raise RuntimeError(f'Interrupted or incomplete training: {directory}')
    write_json(directory / 'complete.json', result)
    print(f'DONE {dataset}/{stage}: {result["exact_performance"]["mean ranking"]}', flush=True)
    return result['checkpoint']


def dataset_pipeline(dataset, gpu, output, smoke=False, controls=True):
    selected = {}
    for stage in CHAIN:
        previous = selected[CHAIN[CHAIN.index(stage)-1]] if stage != 'all' else None
        selected[stage] = run_stage(dataset, stage, gpu, output, previous, smoke)
    if dataset == 'DB15K' and controls:
        for stage in CONTROLS:
            previous = selected['all'] if stage == 'role' else selected['shared'] if stage == 'shared_extended' else None
            selected[stage] = run_stage(dataset, stage, gpu, output, previous, smoke)
    elif controls:
        selected['csprom'] = run_stage(dataset, 'csprom', gpu, output, None, smoke)
    write_json(output / dataset / 'selected_checkpoints.json', selected)
    return selected


def lock_sources(output):
    paths = [ROOT/'main.py', ROOT/'helper.py', ROOT/'kgc_data.py', ROOT/'callbacks.py', ROOT/'locked_callbacks.py']
    paths += list((ROOT/'models').glob('*.py')) + list((ROOT/'script').glob('*.py'))
    paths += list((ROOT/'reference/csprom').glob('*.py')) + [ROOT/'experiments/PROTOCOL.md']
    for dataset in DATASETS:
        paths += list((ROOT/'data/processed'/dataset).glob('*.txt'))
    paths += list((ROOT/'features').glob('*.pkl'))
    hashes = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                digest.update(block)
        hashes[str(path.relative_to(ROOT))] = digest.hexdigest()
    lock = output / 'source_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())['sha256'] != hashes:
            raise RuntimeError('Source or input changed since lock; create a new version.')
    else:
        write_json(lock, dict(version='paper-v1-20260907', seed=41504,
                             created=datetime.datetime.now().isoformat(), sha256=hashes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stage', choices=CHAIN+CONTROLS)
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--previous')
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--no-controls', action='store_true')
    p.add_argument('--skip-final-analysis', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.stage:
        run_stage(args.dataset, args.stage, args.gpu, args.output, args.previous, args.smoke)
        return
    if not args.smoke:
        lock_sources(args.output)
    assignments = [(args.dataset, args.gpu)] if args.dataset else list(zip(DATASETS, [0, 1, 2]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        futures = [pool.submit(dataset_pipeline, d, g, args.output, args.smoke, not args.no_controls) for d,g in assignments]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if not args.skip_final_analysis:
        for dataset, _ in assignments:
            run_command([str(PYTHON), str(ROOT/'script/final_analysis.py'), '--suite', str(args.output),
                         '--dataset', dataset, '--gpu', str(args.gpu)] + (['--smoke'] if args.smoke else []),
                        args.output/dataset/'analysis', args.gpu)


if __name__ == '__main__':
    main()
