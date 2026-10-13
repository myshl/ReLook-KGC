#!/usr/bin/env python3
"""Portable entry points around unmodified historical code snapshots.

No model, loss, checkpoint-selection or ranking logic is changed here.
Only artifact paths and the selected GPU/output directory are supplied anew.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['DB15K', 'YAGO15K', 'FB15k-237']


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(args):
    checksums = read_json(ROOT / 'source_checksums.json')
    for name, expected in checksums.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError('Historical source differs: ' + name)
    if args.assets:
        datasets = [args.dataset] if args.dataset else DATASETS
        inputs = read_json(ROOT / 'input_checksums.json')
        for dataset in datasets:
            for name, expected in inputs[dataset].items():
                if sha256(args.assets / name) != expected:
                    raise RuntimeError('Paper input differs: ' + name)
    print('Checksums verified.')


def environment():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    return env


def stage(args):
    command_file = ROOT / 'configs' / args.dataset / args.stage / 'command.json'
    if not command_file.is_file():
        raise ValueError('No archived training command for this dataset/stage')
    command = read_json(command_file)
    version = 'db15k' if args.dataset == 'DB15K' or args.stage == 'csprom' else 'localplus'
    source = ROOT / 'versions' / version
    command[:2] = [sys.executable, str(source / 'main.py')]

    def replace(flag, value):
        if flag in command:
            command[command.index(flag) + 1] = str(value)
        else:
            command.extend([flag, str(value)])

    replace('-dataset_path', args.assets / 'data/processed')
    replace('-feature_dir', args.assets / 'features')
    replace('-pretrained_model', args.assets / 'model_path/bert-base-uncased')
    replace('-gpu', args.gpu)
    replace('-save_dir', args.output / 'checkpoints')
    replace('-metrics_output', args.output / 'valid.json')
    for flag in ['-continue_path', '-csprom_init_path', '-legacy_init_path']:
        if flag in command and command[command.index(flag) + 1]:
            if args.previous is None:
                raise ValueError('This stage requires --previous with its incoming checkpoint')
            replace(flag, args.previous)
    if any('${' in str(value) for value in command):
        raise ValueError('Unresolved archived path in training command')
    print(json.dumps(command, indent=2), flush=True)
    if args.execute:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Refusing to overwrite an existing run: ' + str(args.output))
        args.output.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=source, env=environment(), check=True)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(args.output))
    manifest = read_json(ROOT / 'release_manifest.json')
    specification = manifest['datasets'][args.dataset]
    expected = specification['checkpoints'][args.variant]['sha256']
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != expected and not args.allow_new_checkpoint:
        raise ValueError('Checkpoint differs from the paper. For a newly trained model, explicitly use --allow-new-checkpoint.')
    verify(args)
    version = 'db15k' if args.variant == 'csprom' else specification['version']
    source = ROOT / 'versions' / version
    sys.path[:0] = [str(source), str(source / 'script')]
    os.environ.update({key: value for key, value in environment().items()
                       if key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM']})
    import torch
    from transformers import BertTokenizer
    from helper import read, read_name, read_file, get_gt
    from kgc_data import KGCDataModule
    from models.relook_model import ReLookKG
    from models.csprom_control import CSPromControl
    from final_analysis import collect, direction_metrics

    torch.set_num_threads(4)
    device = torch.device('cpu' if args.gpu < 0 else f'cuda:{args.gpu}')
    if args.gpu >= 0:
        torch.cuda.set_per_process_memory_fraction(.30, args.gpu)
    # torch.load/pickle assets must come from a trusted source.
    payload = torch.load(args.checkpoint, map_location='cpu')
    config = payload['hyper_parameters']['configs']
    if config.dataset != args.dataset:
        raise ValueError('Checkpoint dataset mismatch')
    config.dataset_path = str(args.assets / 'data/processed')
    config.feature_dir = str(args.assets / 'features')
    config.pretrained_model = str(args.assets / 'model_path/bert-base-uncased')
    config.model_path = str(args.checkpoint)
    config.badcase_report = ''
    config.num_workers = 4
    config.val_batch_size = 128
    if args.variant == 'final':
        deployment = specification['deployment']
        config.selection_policy = 'threshold'
        config.relook_max_blend = deployment['blend']
        config.relook_gate_threshold = deployment['threshold']
        config.relook_tail_evidence_gain_cap = deployment['tail_cap']
        config.relook_head_evidence_gain_cap = deployment['head_cap']
    train, valid, test = [read(config, config.dataset_path, args.dataset, split + '2id.txt')
                          for split in ['train', 'valid', 'test']]
    train_tail, train_head = get_gt(config, train)
    all_tail, all_head = get_gt(config, train + valid + test)
    gt = dict(train_tail_gt=train_tail, train_head_gt=train_head,
              all_tail_gt=all_tail, all_head_gt=all_head)
    names, relations = read_name(config, config.dataset_path, args.dataset)
    texts = dict(ent_names=names, rel_names=relations,
                 ent_descs=read_file(config, config.dataset_path, args.dataset,
                                     'entityid2description.txt', 'desc'))
    cls = CSPromControl if config.model_family == 'csprom' else ReLookKG
    model = cls(config, texts, gt)
    model.load_state_dict(payload['state_dict'], strict=True)
    del payload
    model.eval().to(device)
    dm = KGCDataModule(config, train, valid, test, texts,
                      BertTokenizer.from_pretrained(config.pretrained_model), gt)
    rows = collect(model, dm, gt, args.split)
    result = dict(dataset=args.dataset, split=args.split, variant=args.variant,
                  checkpoint_sha256=checkpoint_hash,
                  historical_checkpoint=(checkpoint_hash == expected),
                  protocol=manifest['evaluation'],
                  metrics=direction_metrics(rows['rank'], rows['role']))
    if args.dataset == 'DB15K':
        result['caveat'] = manifest['db15k_caveat']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('verify', help='Check frozen source and optional external assets')
    check.add_argument('--assets', type=Path)
    check.add_argument('--dataset', choices=DATASETS)
    train = sub.add_parser('stage', help='Replay an archived training command; dry run by default')
    train.add_argument('--stage', required=True)
    train.add_argument('--previous', type=Path)
    train.add_argument('--execute', action='store_true')
    test = sub.add_parser('evaluate', help='Full filtered evaluation with frozen paper settings')
    test.add_argument('--checkpoint', type=Path, required=True)
    test.add_argument('--variant', choices=['final', 'backbone', 'csprom'], default='final')
    test.add_argument('--split', choices=['valid', 'test'], default='test')
    test.add_argument('--allow-new-checkpoint', action='store_true')
    for item in [train, test]:
        item.add_argument('--dataset', choices=DATASETS, required=True)
        item.add_argument('--assets', type=Path, required=True)
        item.add_argument('--output', type=Path, required=True)
        item.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    for name in ['assets', 'output', 'previous', 'checkpoint']:
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    {'verify': verify, 'stage': stage, 'evaluate': evaluate}[args.action](args)


if __name__ == '__main__':
    main()
