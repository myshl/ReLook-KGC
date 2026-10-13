import os
import argparse
import json
from datetime import datetime
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import transformers
from transformers import AutoConfig, BertTokenizer
from models.relook_model import ReLookKG
from kgc_data import KGCDataModule
from helper import get_num, read, read_name, read_file, get_gt
from callbacks import PrintingCallback
from locked_callbacks import RunAudit
from models.csprom_control import CSPromControl
from tuning_utils import prefer_initial


transformers.logging.set_verbosity_error()


def main():
    if configs.gpu != '-1' and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(configs.gpu_memory_fraction, int(configs.gpu))
    Model = CSPromControl if configs.model_family == 'csprom' else ReLookKG
    if configs.save_dir == '':
        if configs.jobid == 'XXXXXXXX':
            configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-' + datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
        else:
            configs.save_dir = os.path.join('./checkpoint', configs.jobid)
    os.makedirs(configs.save_dir, exist_ok=True)
    trainer_params = {
        'max_epochs': configs.epochs,  # 1000
        'logger': False,  # TensorBoardLogger
        'num_sanity_val_steps': 0,  # 2
        'check_val_every_n_epoch': configs.check_val_every_n_epoch,
        'enable_progress_bar': not configs.disable_progress_bar,
        'min_epochs': configs.min_epochs,
    }
    if torch.cuda.is_available() and configs.gpu != '-1':
        trainer_params['devices'] = [int(configs.gpu)]
        trainer_params['accelerator'] = 'gpu'
    else:
        trainer_params['accelerator'] = 'cpu'

    print('devices:', trainer_params.get('devices', 'auto'))
    print('accelerator:', trainer_params['accelerator'])


    if configs.accumulate_grad_batches > 1:
        trainer_params['accumulate_grad_batches'] = configs.accumulate_grad_batches
    if configs.use_fp16:
        trainer_params['precision'] = 16
    if configs.limit_train_batches > 0:
        trainer_params['limit_train_batches'] = configs.limit_train_batches
    if configs.limit_val_batches > 0:
        trainer_params['limit_val_batches'] = configs.limit_val_batches
    if configs.limit_test_batches > 0:
        trainer_params['limit_test_batches'] = configs.limit_test_batches

    ## read triples
    train = read(configs, configs.dataset_path, configs.dataset, 'train2id.txt')
    valid = read(configs, configs.dataset_path, configs.dataset, 'valid2id.txt')
    test = read(configs, configs.dataset_path, configs.dataset, 'test2id.txt')
    all_triples = train + valid + test

    ## construct ground truth dictionary
    # ground truth .shape: dict, example: {hr_str_key1: [t_id11, t_id12, ...], (hr_str_key2: [t_id21, t_id22, ...], ...}
    train_tail_gt, train_head_gt = get_gt(configs, train)
    all_tail_gt, all_head_gt = get_gt(configs, all_triples)

    gt = {
        'train_tail_gt': train_tail_gt,
        'train_head_gt': train_head_gt,
        'all_tail_gt': all_tail_gt,
        'all_head_gt': all_head_gt,
        'relation_frequency': [
            sum(1 for triple in train if triple[2] == relation)
            for relation in range(configs.n_rel)
        ],
    }

    ent_names, rel_names = read_name(configs, configs.dataset_path, configs.dataset)
    ent_descs = read_file(configs, configs.dataset_path, configs.dataset, 'entityid2description.txt', 'desc')
    tok = BertTokenizer.from_pretrained(configs.pretrained_model, add_prefix_space=False)

    text_dict = {
        'ent_names': ent_names,
        'rel_names': rel_names,
        'ent_descs': ent_descs,
    }
    ## construct datamodule
    datamodule = KGCDataModule(configs, train, valid, test, text_dict, tok, gt)
    print('datamodule construction done.', flush=True)

    ## construct trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mrr',
        dirpath=configs.save_dir,
        filename=configs.dataset + '-{epoch:03d}-{' + "val_mrr" + ':.4f}',
        mode='max'
    )
    printing_callback = PrintingCallback()
    trainer_params['callbacks'] = [
        checkpoint_callback,
        printing_callback,
        RunAudit(),
    ]
    if configs.early_stop_patience > 0:
        trainer_params['callbacks'].append(EarlyStopping(
            monitor='val_mrr', mode='max', patience=configs.early_stop_patience,
            min_delta=configs.early_stop_delta, check_finite=True,
        ))
    trainer = pl.Trainer(**trainer_params)

    ## construct model parameters
    kw_args = {
        'text_dict': text_dict,
        'gt': gt,
    }

    if configs.model_path == '':
        if sum(bool(value) for value in (
            configs.continue_path, configs.legacy_init_path, configs.csprom_init_path
        )) > 1:
            raise ValueError('Specify exactly one initializer/continuation source')
        if configs.continue_path == '':
            model = Model(configs, **kw_args)
            if configs.legacy_init_path:
                model.load_legacy_initializer(configs.legacy_init_path)
            if configs.csprom_init_path:
                model.load_csprom_initializer(configs.csprom_init_path)
        else:
            if configs.legacy_init_path:
                raise ValueError(
                    '-legacy_init_path and -continue_path are mutually exclusive'
                )
            model = Model.load_from_checkpoint(configs.continue_path, strict=True, configs=configs, **kw_args)
        print('model construction done.', flush=True)
        trainable_params, non_trainable_params = 0, 0
        for name, params in model.named_parameters():
            if params.requires_grad:
                print('name:', name, 'shape:', params.shape, 'numel:', params.numel())
                trainable_params += params.numel()
            else:
                non_trainable_params += params.numel()
        print('trainable params:', trainable_params, 'non trainable params:', non_trainable_params)
        initial_mrr, initial_path = None, None
        if configs.keep_initial_checkpoint:
            trainer.validate(model, datamodule=datamodule)
            initial_mrr = model.exact_performance['mean ranking']['mrr']
            initial_path = os.path.abspath(os.path.join(configs.save_dir, 'initial.ckpt'))
            trainer.save_checkpoint(initial_path, weights_only=True)
            with open(os.path.join(configs.save_dir, 'initial_validation.json'), 'w') as f:
                json.dump(dict(checkpoint=initial_path, split='valid',
                               exact_performance=model.exact_performance), f, indent=2)
        pl.seed_everything(configs.seed)  # identical sampling seed after construction
        trainer.fit(model, datamodule)
        model_path = checkpoint_callback.best_model_path
        trained_mrr = (float(checkpoint_callback.best_model_score)
                       if checkpoint_callback.best_model_score is not None else None)
        if prefer_initial(initial_mrr, trained_mrr):
            model_path = initial_path
        with open(os.path.join(configs.save_dir, 'checkpoint_selection.json'), 'w') as f:
            json.dump(dict(split='valid', initial_mrr=initial_mrr, trained_mrr=trained_mrr,
                           selected_checkpoint=model_path,
                           selected_initial=model_path == initial_path), f, indent=2)
        if not model_path:
            # Lightning returns from ``fit`` after a graceful KeyboardInterrupt.
            # Do not then interpret the empty checkpoint path as the repository
            # directory and emit a misleading IsADirectoryError.
            print('No checkpoint was produced; skipping post-fit evaluation.', flush=True)
            return
    else:
        model_path = configs.model_path
    print('model_path:', model_path, flush=True)
    model = Model.load_from_checkpoint(model_path, strict=True, configs=configs, **kw_args)
    if configs.eval_split == 'valid':
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule)
    if configs.metrics_output and hasattr(model.history['perf'], 'iterrows'):
        output_path = os.path.abspath(configs.metrics_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        performance = {}
        for index, row in model.history['perf'].iterrows():
            performance[index] = {}
            for column, value in row.items():
                if isinstance(value, str) and value.endswith('%'):
                    value = float(value[:-1]) / 100.0
                else:
                    value = float(value)
                performance[index][column] = value
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump({
                'dataset': configs.dataset,
                'split': configs.eval_split,
                'checkpoint': os.path.abspath(model_path),
                'backbone_prompt_mode': configs.relook_backbone_prompt_mode,
                'backbone_prompt_schedule': configs.relook_backbone_prompt_schedule,
                'performance': performance,
                'exact_performance': getattr(model, 'exact_performance', None),
            }, output_file, indent=2)
        print('metrics_output:', output_path, flush=True)
    elif configs.metrics_output:
        print(
            'Evaluation did not complete; metrics_output was not written.',
            flush=True,
        )


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    transformers.logging.set_verbosity_error()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser()
    parser.add_argument('-csprom_init_path', default='', type=str)
    parser.add_argument('-keep_initial_checkpoint', action='store_true')
    parser.add_argument('-gpu_memory_fraction', default=0.30, type=float)
    parser.add_argument('-model_family', choices=['relook', 'csprom'], default='relook')
    parser.add_argument('-early_stop_patience', type=int, default=6)
    parser.add_argument('-early_stop_delta', type=float, default=0.0001)
    parser.add_argument('-min_epochs', type=int, default=1)
    parser.add_argument('-selection_policy', choices=['threshold', 'none', 'always', 'random', 'entropy', 'margin', 'learned'], default='threshold')
    parser.add_argument('-selection_budget', type=float, default=0.5)

    parser.add_argument('-dataset_path', type=str, default='./data/processed')
    parser.add_argument(
        '-feature_dir', type=str, default='./img_data',
        help='Directory containing <dataset>_{img_features,entity_embed,relation_embed}.pkl',
    )
    parser.add_argument('-jobid', type=str, default='XXXXXXXX')
    parser.add_argument('-dataset', dest='dataset', default='WN18RR',
                        help='Dataset to use, default: InferWiki16k')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')
    parser.add_argument(
        '-metrics_output', type=str, default='',
        help='Optional JSON file receiving the final validation/test metrics',
    )
    parser.add_argument(
        '-disable_progress_bar', action='store_true',
        help='Disable Lightning progress bars for compact experiment logs',
    )

    parser.add_argument('-pretrained_model', type=str, default='bert-base-uncased', help='BERT model name or local path')
    parser.add_argument('-batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')

    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    parser.add_argument('-eval_split', choices=['test', 'valid'], default='test',
                        help='Split used for checkpoint-only evaluation')
    parser.add_argument('-desc_max_length', default=0, type=int, help='')
    parser.add_argument('-prompt_length', default=0, type=int, help='')
    parser.add_argument('-embed_dim', default=0, type=int, help='')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=8, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=16, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-label_smoothing', default=0., type=float, help='Label smoothing')
    parser.add_argument('-use_log_ranks', action='store_true', help='')
    parser.add_argument('-n_lar', default=0, type=int, help='Number of LARs')
    parser.add_argument('-gamma', default=1., type=float, help='Margin loss: margin between pos and neg')
    parser.add_argument('-alpha', default=0., type=float, help='Weight between CE loss and margin loss')
    parser.add_argument('-graph_model', default='conve', type=str, help='[null | transe | distmult | conve | rotate]')
    parser.add_argument('-loss_gamma', default=0., type=float, help='Gamma for score function of transe and rotate')
    parser.add_argument('-lr_scheduler', default='linear', type=str, help='[linear | cosine]')
    parser.add_argument('-max_lar_samples', default=-1, type=int, help='Maximum of negative samples')
    parser.add_argument('-continue_path', dest='continue_path', default='', help='The path for continuing training')
    parser.add_argument(
        '-legacy_init_path', default='', type=str,
        help='Initialize compatible MKG weights from a legacy checkpoint before current-model training',
    )
    parser.add_argument('-accumulate_grad_batches', default=1, type=int, help='')
    parser.add_argument('-check_val_every_n_epoch', default=10, type=int, help='')
    parser.add_argument('-use_speedup', action='store_true', help='')
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-use_fp16', action='store_true', help='')
    parser.add_argument('-prompt_hidden_dim', default=-1, type=int, help='')
    parser.add_argument('-alpha_step', default=0., type=float, help='')
    parser.add_argument('-relook_topk', default=128, type=int,
                        help='Number of base candidates sent to visual verification')
    parser.add_argument(
        '-relook_scope',
        choices=['candidate', 'global', 'directional', 'hybrid'],
        default='candidate',
        help=(
            'Top-K, full-entity, candidate-tail/global-head, or strong-candidate/'
            'weak-global correction'
        ),
    )
    parser.add_argument('-relook_stat_topk', default=20, type=int,
                        help='Number of base scores used to estimate uncertainty')
    parser.add_argument('-relook_gate_threshold', default=0.5, type=float,
                        help='Visual branch invocation threshold during evaluation')
    parser.add_argument(
        '-relook_gate_mode', choices=['legacy', 'utility', 'evidence'],
        default='legacy',
        help=(
            'Legacy uncertainty, pre-evidence rank utility, or post-evidence '
            'visual utility routing'
        ),
    )
    parser.add_argument(
        '-relook_utility_soft_blend', action='store_true',
        help='Scale accepted utility-gate corrections continuously by utility probability',
    )
    parser.add_argument('-relook_min_gate', default=0.1, type=float,
                        help='Minimum visual blend during training to avoid gate collapse')
    parser.add_argument('-relook_router_hidden', default=-1, type=int,
                        help='Hidden dimension of the uncertainty gate')
    parser.add_argument('-relook_base_weight', default=0.5, type=float,
                        help='Weight of the structure-text base loss')
    parser.add_argument('-relook_rerank_weight', default=1.0, type=float,
                        help='Weight of the visually reranked entity loss')
    parser.add_argument('-relook_visual_weight', default=0.2, type=float,
                        help='Weight of the Top-K visual listwise loss')
    parser.add_argument('-relook_budget_weight', default=0.01, type=float,
                        help='Penalty on expected visual invocation rate')
    parser.add_argument('-relook_gate_weight', default=0.0, type=float,
                        help='Weight of supervised visual-utility routing loss')
    parser.add_argument('-relook_target_rate', default=0.5, type=float,
                        help='Soft upper bound for expected visual invocation rate')
    parser.add_argument('-relook_max_blend', default=1.0, type=float,
                        help='Maximum visual correction strength')
    parser.add_argument('-relook_head_blend', default=-1.0, type=float,
                        help='Optional visual correction cap for inverse/head queries')
    parser.add_argument(
        '-relook_tail_evidence_gain_cap', default=-1.0, type=float,
        help=(
            'Evaluation-time upper trust bound on the Joint Evidence Router '
            'residual for direct/tail queries; <0 disables the bound'
        ),
    )
    parser.add_argument(
        '-relook_head_evidence_gain_cap', default=-1.0, type=float,
        help=(
            'Evaluation-time upper trust bound on the Joint Evidence Router '
            'residual for inverse/head queries; <0 disables the bound'
        ),
    )
    parser.add_argument('-relook_global_blend', default=0.05, type=float,
                        help='Weak outside-candidate correction cap in hybrid scope')
    parser.add_argument('-relook_global_head_blend', default=-1.0, type=float,
                        help='Optional hybrid outside-candidate cap for inverse/head queries')
    parser.add_argument('-relook_truth_set_weight', default=0.0, type=float,
                        help='Mixing weight for multi-positive training-answer supervision')
    parser.add_argument('-relook_head_loss_weight', default=1.0, type=float,
                        help='Relative truth-set loss weight for inverse/head queries')
    parser.add_argument('-relook_train_direction', choices=['both', 'head', 'tail'], default='both',
                        help='Training query directions; evaluation always remains bidirectional')
    parser.add_argument('-relook_bypass_relations', default='', type=str,
                        help='Comma-separated directional relation IDs that should keep base scores')
    parser.add_argument(
        '-relook_relation_blends', default='', type=str,
        help=(
            'Comma-separated directional relation_id:blend overrides, for example '
            '299:0.25,458:0.25'
        ),
    )
    parser.add_argument('-relook_utility_margin', default=0.0, type=float,
                        help='Minimum reciprocal-rank gain used as a positive routing target')
    parser.add_argument(
        '-relook_visual_utility_target',
        choices=['raw', 'filtered'],
        default='filtered',
        help=(
            'Build visual-router utility labels from raw ranks or ranks that '
            'filter other known training answers'
        ),
    )
    parser.add_argument('-relook_utility_gain_scale', default=4.0, type=float,
                        help='Extra gate-loss weight for high-impact routing examples')
    parser.add_argument('-relook_gate_only', action='store_true',
                        help='Freeze the trained reasoner and verifier while calibrating the utility gate')
    parser.add_argument(
        '-relook_relation_only', action='store_true',
        help=(
            'Freeze shared modules and tune only directional relation embeddings; '
            'useful for head-only calibration without changing tail predictions'
        ),
    )
    parser.add_argument(
        '-relook_role_adapters', action='store_true',
        help=(
            'Enable a shared Backbone with separate tail/head residual adapters '
            'and a trainable directional-relation router'
        ),
    )
    parser.add_argument(
        '-relook_backbone_prompt_mode',
        choices=['mixed', 'progressive'],
        default='progressive',
        help=(
            'mixed uses visual prompts in lower layers and structural prompts '
            'in upper layers; progressive uses structural prompts throughout'
        ),
    )
    parser.add_argument(
        '-relook_backbone_prompt_schedule',
        choices=['all', 'input', 'early3', 'early6', 'early9', 'late6'],
        default='all',
        help=(
            'Choose which Transformer layers refresh the structural prompt. '
            'Inactive layers retain the prompt state produced by the preceding '
            'layer, keeping prompt tokens and parameter count unchanged. late6 '
            'refreshes only layers 7--12 as an order-matched control for early6.'
        ),
    )
    parser.add_argument(
        '-relook_deep_experts', action='store_true',
        help='Refine upper BERT layers with shared or tail/head deep prompt experts',
    )
    parser.add_argument(
        '-relook_deep_expert_sharing',
        choices=['shared', 'role', 'hierarchical'], default='role',
        help=(
            'shared is the parameter-matched control; role uses separate '
            'tail/head experts; hierarchical keeps a trained shared expert '
            'and learns tail/head residual experts on top'
        ),
    )
    parser.add_argument('-relook_deep_expert_layers', default=6, type=int,
                        help='Number of upper BERT layers refined by deep experts')
    parser.add_argument('-relook_deep_expert_dim', default=64, type=int,
                        help='Bottleneck width of each role deep prompt expert')
    parser.add_argument('-relook_deep_expert_dropout', default=0.1, type=float,
                        help='Dropout inside deep prompt experts')
    parser.add_argument(
        '-relook_joint_evidence_router', action='store_true',
        help=(
            'Refine the visual utility prior using role-expert route, Backbone '
            'confidence, source entity, and expert delta evidence'
        ),
    )
    parser.add_argument(
        '-relook_adapter_only', action='store_true',
        help='Freeze the validated shared Backbone and train only role adapters/router',
    )
    parser.add_argument(
        '-relook_backbone_only', action='store_true',
        help=(
            'Freeze BERT and evidence modules while adapting only prompt, '
            'structure/image embeddings, projection, and graph decoder'
        ),
    )
    parser.add_argument(
        '-relook_visual_only', action='store_true',
        help='Freeze reasoning modules and train only the visual verifier',
    )
    parser.add_argument(
        '-relook_router_only', action='store_true',
        help='Freeze the shared Backbone and role adapters; train only the router',
    )
    parser.add_argument(
        '-relook_query_router_only', action='store_true',
        help=(
            'Freeze a learned relation prior and train only the bounded, '
            'class-balanced query-level gain predictor'
        ),
    )
    parser.add_argument(
        '-relook_monotonic_query_router', action='store_true',
        help=(
            'Use sign-constrained entropy/margin/delta evidence and a bounded '
            'source-conditioned residual; otherwise retain legacy hybrid routing'
        ),
    )
    parser.add_argument(
        '-relook_query_residual_scale', default=0.75, type=float,
        help='Maximum absolute logit correction from the monotonic query router',
    )
    parser.add_argument('-relook_adapter_dim', default=64, type=int,
                        help='Bottleneck dimension of each role adapter')
    parser.add_argument(
        '-relook_adapter_sharing', choices=['role', 'shared'], default='role',
        help='Use separate head/tail adapters or one shared adapter',
    )
    parser.add_argument('-relook_adapter_dropout', default=0.1, type=float,
                        help='Dropout inside role adapters')
    parser.add_argument('-relook_adapter_scale', default=1.0, type=float,
                        help='Maximum residual scale of each role adapter')
    parser.add_argument('-relook_tail_adapter_scale', default=-1.0, type=float,
                        help='Tail-role residual scale; <0 reuses -relook_adapter_scale')
    parser.add_argument('-relook_head_adapter_scale', default=-1.0, type=float,
                        help='Head-role residual scale; <0 reuses -relook_adapter_scale')
    parser.add_argument('-relook_adapter_lr', default=-1.0, type=float,
                        help='Role-adapter learning rate; <=0 reuses -lr')
    parser.add_argument('-relook_role_adapter_weight', default=0.5, type=float,
                        help='Auxiliary full-strength adapter expert loss')
    parser.add_argument('-relook_adapter_norm_weight', default=1e-4, type=float,
                        help='L2 penalty on role-adapter residuals')
    parser.add_argument('-relook_role_router_hidden', default=128, type=int,
                        help='Hidden dimension of the relation-role router')
    parser.add_argument(
        '-relook_role_router_mode',
        choices=['hybrid', 'relation', 'learned', 'full'],
        default='hybrid',
        help=(
            'hybrid combines a relation-role prior with query-level utility; '
            'relation (or legacy alias learned) uses only the relation prior; '
            'full always applies the selected adapter'
        ),
    )
    parser.add_argument('-relook_role_router_dropout', default=0.1, type=float,
                        help='Dropout inside the relation-role router')
    parser.add_argument('-relook_role_router_init', default=0.5, type=float,
                        help='Initial probability of using a role adapter')
    parser.add_argument('-relook_role_router_lr', default=-1.0, type=float,
                        help='Relation-role router learning rate; <=0 reuses -lr')
    parser.add_argument('-relook_role_router_temperature', default=0.5, type=float,
                        help='Temperature of the train-only adapter utility target')
    parser.add_argument('-relook_role_router_margin', default=0.0, type=float,
                        help='NLL improvement required before routing toward an adapter')
    parser.add_argument(
        '-relook_role_router_target', choices=['rank', 'nll'], default='rank',
        help=(
            'Train the hybrid router from filtered reciprocal-rank utility or '
            'from the legacy likelihood utility target'
        ),
    )
    parser.add_argument(
        '-relook_role_router_rank_margin', default=0.0, type=float,
        help='Minimum absolute reciprocal-rank change used as router supervision',
    )
    parser.add_argument('-relook_role_router_weight', default=0.1, type=float,
                        help='Weight of the trainable relation-role routing loss')
    parser.add_argument('-badcase_report', default='', type=str,
                        help='Directory used to export per-query and relation-level diagnostics')
    parser.add_argument('-limit_train_batches', default=-1, type=int,
                        help='Limit training batches for smoke tests; -1 uses all batches')
    parser.add_argument('-limit_val_batches', default=-1, type=int,
                        help='Limit validation batches for smoke tests; -1 uses all batches')
    parser.add_argument('-limit_test_batches', default=-1, type=int,
                        help='Limit test batches for smoke tests; -1 uses all batches')

    parser.add_argument('-visualize_embeddings', action='store_true',
                        help='Generate embedding visualizations during validation')
    parser.add_argument('-embedding_dim_reduction', default='tSNE', type=str, help='')
    parser.add_argument('-img_save_dir', default='./', type=str, help='')


    configs = parser.parse_args()
    n_ent = get_num(configs.dataset_path, configs.dataset, 'entity')
    n_rel = get_num(configs.dataset_path, configs.dataset, 'relation')
    configs.n_ent = n_ent
    configs.n_rel = n_rel
    configs.vocab_size = AutoConfig.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = AutoConfig.from_pretrained(configs.pretrained_model).hidden_size
    configs.is_temporal = 'ICEWS' in configs.dataset
    if configs.prompt_hidden_dim == -1:
        configs.prompt_hidden_dim = configs.embed_dim // 2
    print(configs, flush=True)

    pl.seed_everything(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()
