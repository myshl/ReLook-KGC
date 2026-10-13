import csv
import json
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from helper import GRAPH_MODEL_CLASS, get_loss_fn, get_performance
from tuning_utils import similarity_margin_loss
from models.bert_for_layerwise import BertModelForLayerwise
from models.prompter import Prompter
from models.relook import (
    DeepPromptExpert, EvidenceUtilityGate, JointEvidenceRouter,
    RelationRoleRouter, RoleAdapter, UncertaintyGate, UtilityGate,
    VisualVerifier,
)


class ReLookKG(pl.LightningModule):
    """Reason with structure and text, then inspect images only when uncertain."""

    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.save_hyperparameters(ignore=['text_dict', 'gt'])
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.train_tail_gt = gt['train_tail_gt']
        self.train_head_gt = gt['train_head_gt']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']
        self.relation_frequency = np.asarray(
            gt.get('relation_frequency', [0] * configs.n_rel), dtype=np.int64
        )
        bypass_relations = getattr(configs, 'relook_bypass_relations', '')
        self.bypass_relation_ids = {
            int(value) for value in bypass_relations.split(',') if value.strip()
        }
        relation_blends = getattr(configs, 'relook_relation_blends', '')
        self.relation_blend_overrides = {}
        for item in relation_blends.split(','):
            if not item.strip():
                continue
            relation_id, blend = item.split(':', maxsplit=1)
            relation_id, blend = int(relation_id), float(blend)
            if not 0 <= relation_id < configs.n_rel * 2:
                raise ValueError(f'Invalid directional relation ID: {relation_id}')
            if blend < 0:
                raise ValueError(f'Relation blend must be non-negative: {item}')
            self.relation_blend_overrides[relation_id] = blend

        feature_dir = Path(getattr(configs, 'feature_dir', './img_data')).expanduser()
        if not feature_dir.is_absolute():
            feature_dir = Path(__file__).resolve().parents[1] / feature_dir
        # The released Freebase directory uses ``FB15k-237`` while its feature
        # files use ``FB15K-237``. Resolve that packaging inconsistency locally
        # instead of mutating or duplicating the shared source data.
        feature_dataset = (
            'FB15K-237' if configs.dataset.lower() == 'fb15k-237'
            else configs.dataset
        )
        image_features = torch.as_tensor(
            self._load_feature_pickle(
                feature_dir / f'{feature_dataset}_img_features.pkl'
            ),
            dtype=torch.float32,
        )
        image_features = F.normalize(image_features, p=2, dim=1)
        if image_features.size(1) == 4096:
            image_features = F.avg_pool2d(
                image_features.view(-1, 64, 64), 4, stride=4
            ).flatten(start_dim=1)

        structure_entities = torch.as_tensor(
            self._load_feature_pickle(
                feature_dir / f'{feature_dataset}_entity_embed.pkl'
            ),
            dtype=torch.float32,
        )
        structure_relations = torch.as_tensor(
            self._load_feature_pickle(
                feature_dir / f'{feature_dataset}_relation_embed.pkl'
            ),
            dtype=torch.float32,
        )

        self._validate_feature_shapes(image_features, structure_entities, structure_relations)
        image_mask_path = feature_dir / f'{feature_dataset}_has_image.pkl'
        if image_mask_path.is_file():
            image_mask = torch.as_tensor(
                self._load_feature_pickle(image_mask_path), dtype=torch.bool
            )
            if image_mask.shape != (configs.n_ent,):
                raise ValueError(
                    f'Image coverage mask must have shape ({configs.n_ent},), '
                    f'got {tuple(image_mask.shape)}'
                )
        else:
            # Existing dense datasets predate the explicit coverage mask.
            image_mask = torch.ones(configs.n_ent, dtype=torch.bool)
        self.register_buffer('has_image_mask', image_mask, persistent=False)
        self.img_ent_embed = nn.Embedding.from_pretrained(image_features, freeze=False)
        if image_features.size(1) == configs.embed_dim:
            self.visual_feature_projection = nn.Identity()
        else:
            self.visual_feature_projection = nn.Linear(
                image_features.size(1), configs.embed_dim, bias=False
            )
        self.gat_ent_embed = nn.Embedding.from_pretrained(structure_entities, freeze=False)
        self.gat_rel_embed = nn.Embedding.from_pretrained(structure_relations, freeze=False)
        self.visual_rel_embed = nn.Embedding.from_pretrained(structure_relations.clone(), freeze=False)

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = configs.prompt_length
        self.plm_configs.prompt_hidden_dim = configs.prompt_hidden_dim
        self.plm = BertModelForLayerwise.from_pretrained(configs.pretrained_model)
        if configs.prompt_length > 0:
            for parameter in self.plm.parameters():
                parameter.requires_grad = False

        # Inherited CSProm structural prompting spans all BERT layers.
        self.gat_prompter = Prompter(
            self.plm_configs,
            configs.embed_dim,
            configs.prompt_length,
            num_layers=self.plm_configs.num_hidden_layers,
        )
        self.prompt_fc = nn.Linear(
            configs.prompt_length * self.plm_configs.hidden_size,
            configs.embed_dim,
        )
        self.graph_model = GRAPH_MODEL_CLASS[configs.graph_model](configs)

        router_hidden = configs.relook_router_hidden
        if router_hidden <= 0:
            router_hidden = max(64, configs.embed_dim // 2)
        self.uncertainty_gate = UncertaintyGate(configs.embed_dim, router_hidden)
        self.utility_gate = UtilityGate(
            configs.embed_dim, router_hidden, configs.n_rel * 2
        )
        self.evidence_utility_gate = EvidenceUtilityGate(
            configs.embed_dim, router_hidden, configs.n_rel * 2, stat_dim=12
        )
        self.relation_visuality = nn.Embedding(configs.n_rel * 2, 1)
        nn.init.zeros_(self.relation_visuality.weight)
        self.visual_verifier = VisualVerifier(configs.embed_dim)

        adapter_dim = getattr(configs, 'relook_adapter_dim', 64)
        adapter_dropout = getattr(configs, 'relook_adapter_dropout', 0.1)
        role_router_hidden = getattr(configs, 'relook_role_router_hidden', 128)
        self.tail_role_adapter = RoleAdapter(
            configs.embed_dim, adapter_dim, adapter_dropout
        )
        self.head_role_adapter = RoleAdapter(
            configs.embed_dim, adapter_dim, adapter_dropout
        )
        self.relation_role_router = RelationRoleRouter(
            configs.embed_dim,
            role_router_hidden,
            configs.n_rel * 2,
            getattr(configs, 'relook_role_router_dropout', 0.1),
            getattr(configs, 'relook_role_router_init', 0.5),
        )
        upper_layers = min(
            max(getattr(configs, 'relook_deep_expert_layers', 6), 1),
            self.plm_configs.num_hidden_layers,
        )
        deep_expert_dim = getattr(configs, 'relook_deep_expert_dim', 64)
        expert_kwargs = {
            'embed_dim': configs.embed_dim,
            'model_dim': self.plm_configs.hidden_size,
            'upper_layers': upper_layers,
            'prompt_tokens': configs.prompt_length * 2,
            'dropout': getattr(configs, 'relook_deep_expert_dropout', 0.1),
        }
        # The shared control uses twice the bottleneck width so its parameter
        # budget closely matches the pair of tail/head experts.
        self.shared_deep_prompt_expert = DeepPromptExpert(
            bottleneck_dim=deep_expert_dim * 2, **expert_kwargs
        )
        self.tail_deep_prompt_expert = DeepPromptExpert(
            bottleneck_dim=deep_expert_dim, **expert_kwargs
        )
        self.head_deep_prompt_expert = DeepPromptExpert(
            bottleneck_dim=deep_expert_dim, **expert_kwargs
        )
        self.shared_deep_query_adapter = RoleAdapter(
            configs.embed_dim, adapter_dim * 2, adapter_dropout
        )
        self.joint_evidence_router = JointEvidenceRouter(
            configs.embed_dim,
            role_router_hidden,
            getattr(configs, 'relook_role_router_dropout', 0.1),
        )

        self.loss_fn = get_loss_fn(configs)
        if configs.n_lar > 0:
            self.lar_loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=configs.gamma,
                distance_function=lambda x, y: self.graph_model.score_fn(x, y[0], y[1]),
            )

        self.alpha = 0.0 if configs.alpha_step > 0 else configs.alpha
        self._MASKING_VALUE = -1e4 if configs.use_fp16 else -1e9
        self.history = {'perf': ..., 'loss': []}
        self.inspection_probabilities = []
        self.inspection_decisions = []
        self._configure_trainable_parameters()

    def _selected_deep_expert_modules(self):
        sharing = getattr(
            self.configs, 'relook_deep_expert_sharing', 'role'
        )
        if sharing == 'shared':
            return [
                self.shared_deep_prompt_expert,
                self.shared_deep_query_adapter,
            ]
        return [
            self.tail_deep_prompt_expert,
            self.head_deep_prompt_expert,
            self.tail_role_adapter,
            self.head_role_adapter,
        ]

    @staticmethod
    def _load_feature_pickle(path):
        """Load NumPy feature pickles across NumPy 1.x/2.x module names."""
        # The released MKG-W/MKG-Y features were serialized with NumPy 2.x,
        # while the reproducible project venv uses NumPy 1.23 for PyTorch 1.13.
        # NumPy arrays themselves are compatible; only their private module
        # path changed. Register the old equivalents without changing envs.
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
        with open(path, 'rb') as feature_file:
            return pickle.load(feature_file)

    def _visual_entities(self, entity_ids=None):
        raw_features = (
            self.img_ent_embed.weight
            if entity_ids is None
            else self.img_ent_embed(entity_ids)
        )
        return self.visual_feature_projection(raw_features)

    def load_legacy_initializer(self, checkpoint_path):
        """Warm-start the current design from compatible legacy MKG weights.

        The old and current architectures use different routers, so loading the
        whole checkpoint would silently mix incomparable methods. This method
        transfers only the common structural decoder/embeddings and explicitly
        maps the old structural prompt and visual feature projection.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        source = checkpoint.get('state_dict', checkpoint)
        target = self.state_dict()
        mapped = {}

        common_prefixes = (
            'img_ent_embed.', 'gat_ent_embed.', 'gat_rel_embed.',
            'graph_model.', 'plm.',
        )
        for name, value in source.items():
            if (
                name.startswith(common_prefixes)
                and name in target
                and target[name].shape == value.shape
            ):
                mapped[name] = value

        aliases = {
            'prompt_fc.weight': 'fc.weight',
            'prompt_fc.bias': 'fc.bias',
            'visual_feature_projection.weight': 'scorer_visual_proj.weight',
        }
        for target_name, source_name in aliases.items():
            if (
                source_name in source
                and target_name in target
                and target[target_name].shape == source[source_name].shape
            ):
                mapped[target_name] = source[source_name]

        prompt_pairs = {
            'gat_prompter.seq.0.weight': (
                'prompt_router.structural_prompt.ent_adapter.0.weight',
                'prompt_router.structural_prompt.rel_adapter.0.weight',
            ),
            'gat_prompter.seq.2.weight': (
                'prompt_router.structural_prompt.ent_adapter.2.weight',
                'prompt_router.structural_prompt.rel_adapter.2.weight',
            ),
        }
        for target_name, source_names in prompt_pairs.items():
            if all(name in source for name in source_names):
                value = torch.stack(
                    [source[name] for name in source_names], dim=0
                ).mean(dim=0)
                if target[target_name].shape == value.shape:
                    mapped[target_name] = value

        if (
            'img_rel_embed.weight' in source
            and 'scorer_visual_proj.weight' in source
        ):
            projected_relations = F.linear(
                source['img_rel_embed.weight'],
                source['scorer_visual_proj.weight'],
            )
            if target['visual_rel_embed.weight'].shape == projected_relations.shape:
                mapped['visual_rel_embed.weight'] = F.normalize(
                    projected_relations, p=2, dim=1
                )

        self.load_state_dict(mapped, strict=False)
        print(
            f'legacy initializer loaded {len(mapped)} tensors from '
            f'{checkpoint_path}',
            flush=True,
        )
        del checkpoint, source, target, mapped

    def load_csprom_initializer(self, checkpoint_path):
        """Exact single-backbone transfer; never average or ensemble models."""
        payload = torch.load(checkpoint_path, map_location='cpu')
        source = payload['state_dict']
        target = self.state_dict()
        prefixes = {
            'ent_embed.': 'gat_ent_embed.', 'rel_embed.': 'gat_rel_embed.',
            'prompter.': 'gat_prompter.', 'fc.': 'prompt_fc.',
            'plm.': 'plm.', 'graph_model.': 'graph_model.',
        }
        mapped = {}
        for key, tensor in source.items():
            for old, new in prefixes.items():
                if key.startswith(old):
                    name = new + key[len(old):]
                    if name not in target or target[name].shape != tensor.shape:
                        raise ValueError(f'Incompatible CSProm tensor: {key} -> {name}')
                    mapped[name] = tensor
                    break
            else:
                raise ValueError(f'Unexpected CSProm tensor: {key}')
        expected = {key for key in target if key.startswith(tuple(prefixes.values()))}
        if set(mapped) != expected:
            raise ValueError(f'Incomplete backbone transfer: {expected - set(mapped)}')
        self.load_state_dict(mapped, strict=False)
        print(f'Exact CSProm backbone transfer: {len(mapped)} tensors from {checkpoint_path}', flush=True)

    def _language_adversarial_loss(self, prediction, labels, lars):
        positive = self.gat_ent_embed(labels).unsqueeze(1)
        negative = self.gat_ent_embed(lars).mean(dim=1, keepdim=True)
        positive_bias = self.graph_model.bias[labels].unsqueeze(-1)
        negative_bias = self.graph_model.bias[lars].mean(dim=-1, keepdim=True)
        return similarity_margin_loss(
            self.graph_model.score_fn(prediction, positive, positive_bias),
            self.graph_model.score_fn(prediction, negative, negative_bias),
            self.configs.gamma,
        )

    def _all_deep_expert_modules(self):
        return [
            self.shared_deep_prompt_expert,
            self.tail_deep_prompt_expert,
            self.head_deep_prompt_expert,
            self.shared_deep_query_adapter,
        ]

    def _configure_trainable_parameters(self):
        if getattr(self.configs, 'relook_visual_only', False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (
                self.img_ent_embed,
                self.visual_feature_projection,
                self.visual_rel_embed,
                self.visual_verifier,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            return
        if getattr(self.configs, 'relook_backbone_only', False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (
                self.img_ent_embed,
                self.visual_feature_projection,
                self.gat_ent_embed,
                self.gat_rel_embed,
                self.visual_rel_embed,
                self.gat_prompter,
                self.prompt_fc,
                self.graph_model,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            return
        if getattr(self.configs, 'relook_router_only', False):
            if not getattr(self.configs, 'relook_role_adapters', False):
                raise ValueError('-relook_router_only requires -relook_role_adapters')
            if (
                getattr(self.configs, 'relook_query_router_only', False)
                and not getattr(
                    self.configs, 'relook_monotonic_query_router', False
                )
            ):
                raise ValueError(
                    '-relook_query_router_only requires '
                    '-relook_monotonic_query_router'
                )
            if getattr(
                self.configs, 'relook_role_router_mode', 'hybrid'
            ) == 'full':
                raise ValueError('-relook_router_only requires a learned router mode')
            for parameter in self.parameters():
                parameter.requires_grad = False
            joint_only = getattr(
                self.configs, 'relook_joint_evidence_router', False
            )
            if joint_only:
                for parameter in self.joint_evidence_router.parameters():
                    parameter.requires_grad = True
            else:
                router_mode = getattr(
                    self.configs, 'relook_role_router_mode', 'hybrid'
                )
                relation_prefixes = (
                    'relation_norm.', 'relation_proj.', 'role_embed.',
                    'relation_bias.', 'out.',
                )
                query_prefixes = (
                    'source_norm.', 'source_proj.', 'query_role_embed.',
                    'query_out.', 'query_stat_weight', 'query_bias',
                )
                query_only = getattr(
                    self.configs, 'relook_query_router_only', False
                )
                for name, parameter in (
                    self.relation_role_router.named_parameters()
                ):
                    if query_only:
                        parameter.requires_grad = name.startswith(query_prefixes)
                    else:
                        parameter.requires_grad = (
                            router_mode != 'relation'
                            or name.startswith(relation_prefixes)
                        )
            return
        if getattr(self.configs, 'relook_adapter_only', False):
            if not getattr(self.configs, 'relook_role_adapters', False):
                raise ValueError('-relook_adapter_only requires -relook_role_adapters')
            for parameter in self.parameters():
                parameter.requires_grad = False
            if getattr(self.configs, 'relook_deep_experts', False):
                adapter_modules = self._selected_deep_expert_modules()
            else:
                adapter_modules = [self.tail_role_adapter]
                if getattr(
                    self.configs, 'relook_adapter_sharing', 'role'
                ) == 'role':
                    adapter_modules.append(self.head_role_adapter)
            if getattr(
                self.configs, 'relook_role_router_mode', 'hybrid'
            ) != 'full':
                adapter_modules.append(self.relation_role_router)
            for module in adapter_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
            return
        if getattr(self.configs, 'relook_relation_only', False):
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.gat_rel_embed.weight.requires_grad = True
            return
        if not getattr(self.configs, 'relook_gate_only', False):
            deep_enabled = getattr(self.configs, 'relook_deep_experts', False)
            selected_ids = {
                id(module) for module in self._selected_deep_expert_modules()
            } if deep_enabled else set()
            for module in self._all_deep_expert_modules() + [
                self.tail_role_adapter, self.head_role_adapter,
            ]:
                if id(module) not in selected_ids:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
            if not getattr(
                self.configs, 'relook_joint_evidence_router', False
            ):
                for parameter in self.joint_evidence_router.parameters():
                    parameter.requires_grad = False
            return
        for parameter in self.parameters():
            parameter.requires_grad = False
        gate = (
            self.evidence_utility_gate
            if getattr(self.configs, 'relook_gate_mode', 'legacy') == 'evidence'
            else self.utility_gate
        )
        for parameter in gate.parameters():
            parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self.configs, 'relook_visual_only', False):
            for module in (
                self.plm, self.gat_prompter, self.prompt_fc, self.graph_model,
                self.gat_ent_embed, self.gat_rel_embed,
                self.uncertainty_gate, self.utility_gate,
                self.evidence_utility_gate, self.relation_visuality,
                self.tail_role_adapter, self.head_role_adapter,
                self.relation_role_router,
                self.shared_deep_prompt_expert,
                self.tail_deep_prompt_expert,
                self.head_deep_prompt_expert,
                self.shared_deep_query_adapter,
                self.joint_evidence_router,
            ):
                module.eval()
            for module in (
                self.img_ent_embed,
                self.visual_feature_projection,
                self.visual_rel_embed,
                self.visual_verifier,
            ):
                module.train(True)
        if mode and getattr(self.configs, 'relook_backbone_only', False):
            for module in (
                self.plm,
                self.uncertainty_gate,
                self.utility_gate,
                self.evidence_utility_gate,
                self.relation_visuality,
                self.visual_verifier,
                self.tail_role_adapter,
                self.head_role_adapter,
                self.relation_role_router,
                self.shared_deep_prompt_expert,
                self.tail_deep_prompt_expert,
                self.head_deep_prompt_expert,
                self.shared_deep_query_adapter,
                self.joint_evidence_router,
            ):
                module.eval()
            for module in (
                self.img_ent_embed,
                self.visual_feature_projection,
                self.gat_ent_embed,
                self.gat_rel_embed,
                self.visual_rel_embed,
                self.gat_prompter,
                self.prompt_fc,
                self.graph_model,
            ):
                module.train(True)
        if mode and getattr(self.configs, 'relook_router_only', False):
            # Router utility labels must refer to a fixed expert. Keep both the
            # shared Backbone and trained role adapters deterministic.
            for module in (
                self.plm, self.gat_prompter, self.prompt_fc, self.graph_model,
                self.uncertainty_gate, self.utility_gate,
                self.evidence_utility_gate, self.relation_visuality,
                self.visual_verifier, self.img_ent_embed, self.gat_ent_embed,
                self.gat_rel_embed, self.visual_rel_embed,
                self.tail_role_adapter, self.head_role_adapter,
                self.shared_deep_prompt_expert,
                self.tail_deep_prompt_expert,
                self.head_deep_prompt_expert,
                self.shared_deep_query_adapter,
                self.joint_evidence_router,
            ):
                module.eval()
            joint_only = getattr(
                self.configs, 'relook_joint_evidence_router', False
            )
            self.relation_role_router.train(not joint_only)
            if (
                not joint_only
                and getattr(self.configs, 'relook_query_router_only', False)
            ):
                # The relation prior is frozen in the cumulative query-stage
                # ablation; in particular, its Dropout must also stay frozen.
                self.relation_role_router.out.eval()
            self.joint_evidence_router.train(
                joint_only
            )
        if mode and getattr(self.configs, 'relook_adapter_only', False):
            # A frozen Backbone must also keep dropout/BatchNorm deterministic;
            # only the lightweight role modules remain in training mode.
            for module in (
                self.plm, self.gat_prompter, self.prompt_fc, self.graph_model,
                self.uncertainty_gate, self.utility_gate,
                self.evidence_utility_gate, self.relation_visuality,
                self.visual_verifier, self.img_ent_embed, self.gat_ent_embed,
                self.gat_rel_embed, self.visual_rel_embed,
            ):
                module.eval()
            self.tail_role_adapter.train(True)
            self.head_role_adapter.train(
                getattr(self.configs, 'relook_adapter_sharing', 'role') == 'role'
            )
            self.relation_role_router.train(
                getattr(
                    self.configs, 'relook_role_router_mode', 'hybrid'
                ) != 'full'
            )
            for module in self._all_deep_expert_modules():
                module.train(False)
            if getattr(self.configs, 'relook_deep_experts', False):
                for module in self._selected_deep_expert_modules():
                    module.train(True)
        if mode and getattr(self.configs, 'relook_relation_only', False):
            # Dropout or BatchNorm updates in frozen shared modules would still
            # alter tail predictions. Keep the full path deterministic while
            # optimizing only the independently indexed relation embeddings.
            for module in (
                self.plm, self.gat_prompter, self.prompt_fc, self.graph_model,
                self.uncertainty_gate, self.utility_gate,
                self.evidence_utility_gate,
                self.relation_visuality, self.visual_verifier,
                self.img_ent_embed, self.gat_ent_embed, self.visual_rel_embed,
                self.tail_role_adapter, self.head_role_adapter,
                self.relation_role_router,
                self.shared_deep_prompt_expert,
                self.tail_deep_prompt_expert,
                self.head_deep_prompt_expert,
                self.shared_deep_query_adapter,
                self.joint_evidence_router,
            ):
                module.eval()
            self.gat_rel_embed.train(True)
        if mode and getattr(self.configs, 'relook_gate_only', False):
            # Keep dropout and BatchNorm statistics of the validated checkpoint fixed.
            for module in (
                self.plm, self.gat_prompter, self.prompt_fc, self.graph_model,
                self.uncertainty_gate, self.utility_gate,
                self.evidence_utility_gate, self.relation_visuality,
                self.visual_verifier,
                self.img_ent_embed, self.visual_feature_projection,
                self.gat_ent_embed, self.gat_rel_embed,
                self.visual_rel_embed,
                self.tail_role_adapter, self.head_role_adapter,
                self.relation_role_router,
                self.shared_deep_prompt_expert,
                self.tail_deep_prompt_expert,
                self.head_deep_prompt_expert,
                self.shared_deep_query_adapter,
                self.joint_evidence_router,
            ):
                module.eval()
            gate = (
                self.evidence_utility_gate
                if getattr(self.configs, 'relook_gate_mode', 'legacy') == 'evidence'
                else self.utility_gate
            )
            gate.train(True)
        return self

    def _validate_feature_shapes(self, images, entities, relations):
        expected_relations = self.configs.n_rel * 2
        if images.ndim != 2 or images.size(0) != self.configs.n_ent:
            raise ValueError(
                f'Image features must be a matrix with {self.configs.n_ent} rows, '
                f'got {tuple(images.shape)}'
            )
        if entities.shape != (self.configs.n_ent, self.configs.embed_dim):
            raise ValueError(
                f'Structure entity features must have shape '
                f'({self.configs.n_ent}, {self.configs.embed_dim}), got {tuple(entities.shape)}'
            )
        if relations.shape != (expected_relations, self.configs.embed_dim):
            raise ValueError(
                f'Structure relation features must have shape '
                f'({expected_relations}, {self.configs.embed_dim}), got {tuple(relations.shape)}'
            )

    def _shared_backbone_forward(
        self, ent_rel, src_ids, src_mask, layerwise_prompt_delta=None
    ):
        ent_ids, rel_ids = ent_rel[:, 0], ent_rel[:, 1]
        structure_ent = self.gat_ent_embed(ent_ids)
        structure_rel = self.gat_rel_embed(rel_ids)
        layerwise_prompt = self.gat_prompter(
            torch.stack([structure_ent, structure_rel], dim=1)
        )
        backbone_mode = getattr(
            self.configs, 'relook_backbone_prompt_mode', 'progressive'
        )
        prompt_refresh_mask = None
        if backbone_mode == 'mixed':
            # Multimodal mixed control (not official CSProm): early layers receive visual
            # prompts and later layers receive structural prompts.  Reusing the
            # same generator isolates evidence ordering from raw capacity.
            image_prompt = self.gat_prompter(torch.stack([
                self._visual_entities(ent_ids),
                self.visual_rel_embed(rel_ids),
            ], dim=1))
            split_layer = self.plm_configs.num_hidden_layers // 2
            layerwise_prompt = torch.cat([
                image_prompt[:, :split_layer],
                layerwise_prompt[:, split_layer:],
            ], dim=1)
        elif backbone_mode == 'progressive':
            schedule = getattr(
                self.configs, 'relook_backbone_prompt_schedule', 'all'
            )
            layer_count = self.plm_configs.num_hidden_layers
            schedule_masks = {
                'all': [True] * layer_count,
                'input': [index == 0 for index in range(layer_count)],
                'early3': [index < min(3, layer_count) for index in range(layer_count)],
                'early6': [index < min(6, layer_count) for index in range(layer_count)],
                'early9': [index < min(9, layer_count) for index in range(layer_count)],
                # Order-matched control: use the same six refresh operations as
                # early6, but move them from the lower to the upper half.
                'late6': [
                    index >= layer_count // 2
                    for index in range(layer_count)
                ],
            }
            if schedule not in schedule_masks:
                raise ValueError(
                    f'Unsupported Backbone prompt schedule: {schedule}'
                )
            prompt_refresh_mask = schedule_masks[schedule]
        else:
            raise ValueError(
                f'Unsupported Backbone prompt mode: {backbone_mode}'
            )

        if layerwise_prompt_delta is not None:
            upper_layers = layerwise_prompt_delta.size(1)
            if upper_layers > layerwise_prompt.size(1):
                raise ValueError(
                    'Deep expert has more layers than the shared prompt'
                )
            layerwise_prompt = layerwise_prompt.clone()
            layerwise_prompt[:, -upper_layers:] = (
                layerwise_prompt[:, -upper_layers:]
                + layerwise_prompt_delta
            )

        prompt_mask = torch.ones(
            src_ids.size(0), self.configs.prompt_length * 2,
            dtype=src_mask.dtype, device=src_mask.device,
        )
        attention_mask = torch.cat([prompt_mask, src_mask], dim=1)
        output = self.plm(
            input_ids=src_ids,
            attention_mask=attention_mask,
            layerwise_prompt=layerwise_prompt,
            layerwise_prompt_mask=prompt_refresh_mask,
        )

        prompt_state = output.last_hidden_state[:, :self.configs.prompt_length * 2]
        entity_state, relation_state = torch.chunk(prompt_state, chunks=2, dim=1)
        entity_state = self.prompt_fc(entity_state.flatten(start_dim=1))
        relation_state = self.prompt_fc(relation_state.flatten(start_dim=1))
        prediction = self.graph_model(entity_state, relation_state)
        shared_logits = self.graph_model.get_logits(
            prediction, self.gat_ent_embed.weight
        )
        return (
            shared_logits, prediction, structure_ent, structure_rel,
            layerwise_prompt,
        )

    def _deep_prompt_delta(self, structure_ent, structure_rel, roles):
        sharing = getattr(
            self.configs, 'relook_deep_expert_sharing', 'role'
        )
        if sharing == 'shared':
            return self.shared_deep_prompt_expert(
                structure_ent, structure_rel
            )
        tail_delta = self.tail_deep_prompt_expert(
            structure_ent, structure_rel
        )
        head_delta = self.head_deep_prompt_expert(
            structure_ent, structure_rel
        )
        role_shape = [roles.size(0)] + [1] * (tail_delta.ndim - 1)
        return torch.where(roles.view(*role_shape).bool(), head_delta, tail_delta)

    def _base_forward(self, ent_rel, src_ids, src_mask):
        freeze_backbone = (
            getattr(self.configs, 'relook_adapter_only', False)
            or getattr(self.configs, 'relook_router_only', False)
            or getattr(self.configs, 'relook_visual_only', False)
        )
        context = torch.no_grad() if freeze_backbone else nullcontext()
        with context:
            (
                shared_logits, shared_prediction, structure_ent,
                structure_rel, _,
            ) = (
                self._shared_backbone_forward(ent_rel, src_ids, src_mask)
            )

        if not getattr(self.configs, 'relook_role_adapters', False):
            zeros = shared_prediction.new_zeros(shared_prediction.size(0))
            return {
                'base_logits': shared_logits,
                'shared_logits': shared_logits,
                'adapter_logits': shared_logits,
                'prediction': shared_prediction,
                'shared_prediction': shared_prediction,
                'structure_ent': structure_ent,
                'structure_rel': structure_rel,
                'role_router_probability': zeros,
                'role_router_prior_probability': zeros,
                'role_router_query_gain_logit': zeros,
                'role_router_query_probability': zeros.new_full(
                    zeros.shape, 0.5
                ),
                'role_router_entropy': zeros,
                'role_router_margin': zeros,
                'role_router_delta_norm': zeros,
                'role_adapter_delta': torch.zeros_like(shared_prediction),
            }

        roles = (ent_rel[:, 1] >= self.configs.n_rel).long()
        deep_experts = getattr(self.configs, 'relook_deep_experts', False)
        if deep_experts:
            sharing = getattr(
                self.configs, 'relook_deep_expert_sharing', 'role'
            )
            if sharing == 'hierarchical':
                # Stage 1 is the already trained shared upper-layer Expert.
                # It becomes the Router's fallback path.  Stage 2 adds only a
                # tail/head prompt and query residual, so role specialization
                # cannot discard the shared solution learned from all queries.
                shared_prompt_delta = self.shared_deep_prompt_expert(
                    structure_ent, structure_rel
                )
                with torch.no_grad():
                    (
                        _, expert_shared_deep_prediction, _, _, _,
                    ) = self._shared_backbone_forward(
                        ent_rel,
                        src_ids,
                        src_mask,
                        layerwise_prompt_delta=shared_prompt_delta,
                    )
                    expert_shared_prediction = (
                        expert_shared_deep_prediction
                        + self.shared_deep_query_adapter(
                            expert_shared_deep_prediction, structure_rel
                        )
                    )

                role_prompt_delta = self._deep_prompt_delta(
                    structure_ent, structure_rel, roles
                )
                (
                    _, deep_prediction, _, _, _,
                ) = self._shared_backbone_forward(
                    ent_rel,
                    src_ids,
                    src_mask,
                    layerwise_prompt_delta=(
                        shared_prompt_delta + role_prompt_delta
                    ),
                )
                common_query_delta = self.shared_deep_query_adapter(
                    deep_prediction, structure_rel
                )
                common_prediction = deep_prediction + common_query_delta
                tail_query_delta = self.tail_role_adapter(
                    common_prediction, structure_rel
                )
                head_query_delta = self.head_role_adapter(
                    common_prediction, structure_rel
                )
                query_delta = torch.where(
                    roles.unsqueeze(1).bool(),
                    head_query_delta,
                    tail_query_delta,
                )
                role_prediction = common_prediction + query_delta
                shared_prediction = expert_shared_prediction
                shared_logits = self.graph_model.get_logits(
                    shared_prediction, self.gat_ent_embed.weight
                )
                role_delta = role_prediction - shared_prediction
            else:
                prompt_delta = self._deep_prompt_delta(
                    structure_ent, structure_rel, roles
                )
                (
                    _, deep_prediction, _, _, _,
                ) = self._shared_backbone_forward(
                    ent_rel,
                    src_ids,
                    src_mask,
                    layerwise_prompt_delta=prompt_delta,
                )
            if sharing == 'shared':
                query_delta = self.shared_deep_query_adapter(
                    deep_prediction, structure_rel
                )
            elif sharing == 'role':
                tail_query_delta = self.tail_role_adapter(
                    deep_prediction, structure_rel
                )
                head_query_delta = self.head_role_adapter(
                    deep_prediction, structure_rel
                )
                query_delta = torch.where(
                    roles.unsqueeze(1).bool(),
                    head_query_delta,
                    tail_query_delta,
                )
            if sharing != 'hierarchical':
                role_delta = (
                    deep_prediction + query_delta
                ) - shared_prediction
        else:
            tail_delta = self.tail_role_adapter(
                shared_prediction, structure_rel
            )
            if getattr(
                self.configs, 'relook_adapter_sharing', 'role'
            ) == 'shared':
                head_delta = tail_delta
            else:
                head_delta = self.head_role_adapter(
                    shared_prediction, structure_rel
                )
            role_delta = torch.where(
                roles.unsqueeze(1).bool(), head_delta, tail_delta
            )
        adapter_scale = getattr(self.configs, 'relook_adapter_scale', 1.0)
        tail_scale = getattr(
            self.configs, 'relook_tail_adapter_scale', -1.0
        )
        head_scale = getattr(
            self.configs, 'relook_head_adapter_scale', -1.0
        )
        tail_scale = adapter_scale if tail_scale < 0 else tail_scale
        head_scale = adapter_scale if head_scale < 0 else head_scale
        role_scale = torch.where(
            roles.bool(),
            role_delta.new_full(roles.shape, head_scale),
            role_delta.new_full(roles.shape, tail_scale),
        )
        role_delta = role_scale.unsqueeze(1) * role_delta
        adapter_prediction = shared_prediction + role_delta
        adapter_logits = self.graph_model.get_logits(
            adapter_prediction, self.gat_ent_embed.weight
        )
        router_mode = getattr(
            self.configs, 'relook_role_router_mode', 'hybrid'
        )
        if router_mode == 'full':
            route_probability = shared_prediction.new_ones(
                shared_prediction.size(0)
            )
            router_details = {
                'prior_probability': route_probability,
                'query_gain_logit': shared_prediction.new_zeros(
                    shared_prediction.size(0)
                ),
                'query_utility_probability': shared_prediction.new_full(
                    (shared_prediction.size(0),), 0.5
                ),
                'query_entropy': shared_prediction.new_zeros(
                    shared_prediction.size(0)
                ),
                'query_margin': shared_prediction.new_zeros(
                    shared_prediction.size(0)
                ),
                'adapter_delta_norm': shared_prediction.new_zeros(
                    shared_prediction.size(0)
                ),
            }
        else:
            route_probability, router_details = self.relation_role_router(
                structure_rel,
                ent_rel[:, 1],
                roles,
                source_entity=structure_ent,
                shared_logits=shared_logits,
                shared_query=shared_prediction,
                adapter_delta=role_delta,
                stat_topk=getattr(self.configs, 'relook_stat_topk', 20),
                # ``learned`` remains a legacy alias for relation-only routing
                # so existing checkpoints retain their original behavior.
                use_query_utility=router_mode == 'hybrid',
                monotonic_query_utility=getattr(
                    self.configs, 'relook_monotonic_query_router', False
                ),
                query_residual_scale=getattr(
                    self.configs, 'relook_query_residual_scale', 0.75
                ),
                return_details=True,
            )
        prediction = shared_prediction + route_probability.unsqueeze(1) * role_delta
        base_logits = self.graph_model.get_logits(
            prediction, self.gat_ent_embed.weight
        )
        return {
            'base_logits': base_logits,
            'shared_logits': shared_logits,
            'adapter_logits': adapter_logits,
            'prediction': prediction,
            'shared_prediction': shared_prediction,
            'structure_ent': structure_ent,
            'structure_rel': structure_rel,
            'role_router_probability': route_probability,
            'role_router_prior_probability': router_details['prior_probability'],
            'role_router_query_gain_logit': router_details['query_gain_logit'],
            'role_router_query_probability': router_details[
                'query_utility_probability'
            ],
            'role_router_entropy': router_details['query_entropy'],
            'role_router_margin': router_details['query_margin'],
            'role_router_delta_norm': router_details['adapter_delta_norm'],
            'role_adapter_delta': role_delta,
        }

    def _candidate_ids(self, base_logits, target_ids=None):
        candidate_count = min(self.configs.relook_topk, self.configs.n_ent)
        candidate_ids = torch.topk(base_logits.detach(), k=candidate_count, dim=1).indices
        if target_ids is None:
            return candidate_ids

        contains_target = candidate_ids.eq(target_ids.unsqueeze(1)).any(dim=1)
        if not contains_target.all():
            candidate_ids = candidate_ids.clone()
            candidate_ids[~contains_target, -1] = target_ids[~contains_target]
        return candidate_ids

    def _visual_scores(self, ent_rel, candidate_ids):
        ent_ids, rel_ids = ent_rel[:, 0], ent_rel[:, 1]
        head_images = self._visual_entities(ent_ids)
        candidate_images = self._visual_entities(candidate_ids)
        visual_relations = self.visual_rel_embed(rel_ids)
        return self.visual_verifier(head_images, visual_relations, candidate_images)

    def _blend_cap(self, relation_ids, reference):
        """Use a separately calibrated correction strength for inverse queries."""
        caps = reference.new_full(
            reference.shape, getattr(self.configs, 'relook_max_blend', 1.0)
        )
        head_blend = getattr(self.configs, 'relook_head_blend', -1.0)
        if head_blend >= 0:
            caps = torch.where(relation_ids >= self.configs.n_rel, head_blend, caps)
        for relation_id, relation_blend in self.relation_blend_overrides.items():
            caps = torch.where(
                relation_ids == relation_id,
                caps.new_full(caps.shape, relation_blend),
                caps,
            )
        return caps

    def _global_blend_cap(self, relation_ids, reference):
        """Return the weak rescue strength used outside the trusted shortlist."""
        caps = reference.new_full(
            reference.shape, getattr(self.configs, 'relook_global_blend', 0.05)
        )
        head_blend = getattr(self.configs, 'relook_global_head_blend', -1.0)
        if head_blend >= 0:
            caps = torch.where(relation_ids >= self.configs.n_rel, head_blend, caps)
        return caps

    def _inference_blend(self, caps, probabilities):
        """Keep utility routing discrete unless soft calibration is requested."""
        is_utility = getattr(
            self.configs, 'relook_gate_mode', 'legacy'
        ) in ('utility', 'evidence')
        if is_utility and not getattr(
            self.configs, 'relook_utility_soft_blend', False
        ):
            return caps
        return caps * probabilities

    def _role_aware_evidence_trust_region(
        self, visual_prior_probability, gain_logit, relation_ids
    ):
        """Bound over-confident positive evidence residuals at evaluation.

        The joint Router is trained without clipping so its utility objective
        retains a useful gradient.  At inference, role-specific upper bounds
        keep its source-conditioned residual from overwhelming the separately
        learned visual prior.  Negative residuals remain untouched because
        they safely suppress unnecessary verification.
        """
        effective_gain = gain_logit
        if self.training:
            probability = torch.sigmoid(
                JointEvidenceRouter._logit(visual_prior_probability)
                + effective_gain
            )
            return probability, effective_gain

        tail_cap = getattr(
            self.configs, 'relook_tail_evidence_gain_cap', -1.0
        )
        head_cap = getattr(
            self.configs, 'relook_head_evidence_gain_cap', -1.0
        )
        if tail_cap >= 0 or head_cap >= 0:
            effective_gain = gain_logit.clone()
            tail_mask = relation_ids < self.configs.n_rel
            if tail_cap >= 0:
                effective_gain[tail_mask] = effective_gain[tail_mask].clamp_max(
                    tail_cap
                )
            if head_cap >= 0:
                effective_gain[~tail_mask] = effective_gain[~tail_mask].clamp_max(
                    head_cap
                )
        probability = torch.sigmoid(
            JointEvidenceRouter._logit(visual_prior_probability)
            + effective_gain
        )
        return probability, effective_gain

    def _evidence_statistics(
        self, base_logits, candidate_ids, visual_scores, relation_ids
    ):
        """Summarize visual agreement without consulting the target entity."""
        base_stats = UtilityGate.confidence_statistics(
            base_logits, self.configs.relook_stat_topk
        )
        candidate_base = base_logits.gather(1, candidate_ids).detach().float()
        visual = visual_scores.detach().float()
        candidate_count = visual.size(1)

        visual_probs = F.softmax(visual, dim=1)
        visual_entropy = -(
            visual_probs * torch.log(visual_probs.clamp_min(1e-12))
        ).sum(dim=1) / np.log(max(candidate_count, 2))
        visual_top = torch.topk(visual, k=min(2, candidate_count), dim=1).values
        visual_scale = visual.std(dim=1, unbiased=False).clamp_min(1e-6)
        if candidate_count > 1:
            visual_margin = torch.tanh(
                (visual_top[:, 0] - visual_top[:, 1]) / visual_scale / 5.0
            )
        else:
            visual_margin = torch.zeros_like(visual_entropy)
        visual_winner = visual.argmax(dim=1)

        caps = self._blend_cap(relation_ids, visual[:, 0])
        corrected = candidate_base + caps.unsqueeze(1) * visual
        corrected_top = torch.topk(
            corrected, k=min(2, candidate_count), dim=1
        ).values
        corrected_scale = corrected.std(dim=1, unbiased=False).clamp_min(1e-6)
        if candidate_count > 1:
            corrected_margin = torch.tanh(
                (corrected_top[:, 0] - corrected_top[:, 1])
                / corrected_scale / 5.0
            )
        else:
            corrected_margin = torch.zeros_like(visual_entropy)
        corrected_winner = corrected.argmax(dim=1)
        visual_advantage = torch.tanh(
            (visual_top[:, 0] - visual[:, 0]) / visual_scale / 5.0
        )

        evidence_stats = torch.stack(
            [
                visual_entropy,
                visual_probs.max(dim=1).values,
                visual_margin,
                visual_probs[:, 0],
                visual_winner.ne(0).to(visual.dtype),
                corrected_winner.ne(0).to(visual.dtype),
                corrected_margin,
                visual_advantage,
            ],
            dim=1,
        )
        return torch.cat([base_stats, evidence_stats], dim=1)

    def _truth_set_loss(
        self, logits, ent_rel, fallback_labels, reduction='mean'
    ):
        """Cross entropy against every known training answer of the query.

        YAGO inverse queries are often one-to-many. A single-label softmax makes
        those true entities compete with one another; the uniform truth-set
        target removes that false-negative pressure without using validation or
        test triples.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        row_ids = []
        answer_ids = []
        queries = ent_rel.detach().cpu().tolist()
        fallback = fallback_labels.detach().cpu().tolist()
        for row, ((entity_id, relation_id), label) in enumerate(zip(queries, fallback)):
            ground_truth = (
                self.train_tail_gt if relation_id < self.configs.n_rel
                else self.train_head_gt
            )
            answers = ground_truth.get((entity_id, relation_id), [label])
            # De-duplicate repeated triples while retaining deterministic order.
            answers = list(dict.fromkeys(answers)) or [label]
            row_ids.extend([row] * len(answers))
            answer_ids.extend(answers)

        row_tensor = torch.as_tensor(row_ids, dtype=torch.long, device=logits.device)
        answer_tensor = torch.as_tensor(
            answer_ids, dtype=torch.long, device=logits.device
        )
        positive_log_probs = log_probs[row_tensor, answer_tensor]
        positive_sum = positive_log_probs.new_zeros(logits.size(0)).scatter_add_(
            0, row_tensor, positive_log_probs
        )
        positive_count = torch.bincount(
            row_tensor, minlength=logits.size(0)
        ).to(positive_log_probs.dtype).clamp_min(1.0)
        positive_nll = -positive_sum / positive_count

        smoothing = self.configs.label_smoothing
        per_query_loss = (
            (1.0 - smoothing) * positive_nll
            - smoothing * log_probs.mean(dim=1)
        )
        head_weight = getattr(self.configs, 'relook_head_loss_weight', 1.0)
        weights = torch.where(
            ent_rel[:, 1] >= self.configs.n_rel,
            per_query_loss.new_full(per_query_loss.shape, head_weight),
            per_query_loss.new_ones(per_query_loss.shape),
        )
        if reduction == 'none':
            return per_query_loss
        if reduction != 'mean':
            raise ValueError(f'Unsupported truth-set reduction: {reduction}')
        return (per_query_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _training_filtered_reciprocal_rank(self, logits, ent_rel, labels):
        """Compute target ranks after filtering other known training answers."""
        with torch.no_grad():
            detached_logits = logits.detach()
            target_scores = detached_logits.gather(
                1, labels.unsqueeze(1)
            ).squeeze(1)
            unfiltered_higher = detached_logits.gt(
                target_scores.unsqueeze(1)
            ).sum(dim=1)
            queries = ent_rel.detach().cpu().tolist()
            fallback = labels.detach().cpu().tolist()
            row_ids = []
            answer_ids = []
            for row, ((entity_id, relation_id), label) in enumerate(
                zip(queries, fallback)
            ):
                ground_truth = (
                    self.train_tail_gt if relation_id < self.configs.n_rel
                    else self.train_head_gt
                )
                answers = ground_truth.get((entity_id, relation_id), [label])
                answers = list(dict.fromkeys(answers)) or [label]
                row_ids.extend([row] * len(answers))
                answer_ids.extend(answers)

            row_tensor = torch.as_tensor(
                row_ids, dtype=torch.long, device=logits.device
            )
            answer_tensor = torch.as_tensor(
                answer_ids, dtype=torch.long, device=logits.device
            )
            filtered_answer_higher = detached_logits[
                row_tensor, answer_tensor
            ].gt(target_scores[row_tensor]).to(unfiltered_higher.dtype)
            filtered_higher_count = unfiltered_higher.new_zeros(
                logits.size(0)
            ).scatter_add_(0, row_tensor, filtered_answer_higher)
            ranks = 1 + unfiltered_higher - filtered_higher_count
            return ranks.to(logits.dtype).reciprocal()

    @staticmethod
    def _raw_reciprocal_rank(logits, labels):
        """Compute target reciprocal ranks without filtering other answers."""
        with torch.no_grad():
            detached_logits = logits.detach()
            target_scores = detached_logits.gather(
                1, labels.unsqueeze(1)
            ).squeeze(1)
            ranks = 1 + detached_logits.gt(
                target_scores.unsqueeze(1)
            ).sum(dim=1)
            return ranks.to(logits.dtype).reciprocal()

    @staticmethod
    def _amp_safe_binary_cross_entropy(
        probabilities, targets, weight=None, reduction='mean'
    ):
        """Evaluate probability-space BCE in FP32 under mixed precision.

        PyTorch intentionally rejects ``binary_cross_entropy`` while CUDA
        autocast is active because its backward pass can underflow in FP16.
        Routers expose calibrated probabilities rather than raw logits, so the
        numerically equivalent safe choice is a small autocast-disabled FP32
        region instead of changing their public interface.
        """
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy(
                probabilities.float(),
                targets.float(),
                weight=None if weight is None else weight.float(),
                reduction=reduction,
            )

    def _rank_utility_router_loss(self, output, ent_rel, labels):
        """Supervise routing only where the adapter changes filtered RR."""
        shared_rr = self._training_filtered_reciprocal_rank(
            output['shared_logits'], ent_rel, labels
        )
        adapter_rr = self._training_filtered_reciprocal_rank(
            output['adapter_logits'], ent_rel, labels
        )
        rank_gain = adapter_rr - shared_rr
        margin = getattr(
            self.configs, 'relook_role_router_rank_margin', 0.0
        )
        informative = rank_gain.abs() > margin
        targets = rank_gain.gt(margin).to(output['role_router_probability'].dtype)
        per_query_loss = self._amp_safe_binary_cross_entropy(
            (
                output['role_router_query_probability']
                if getattr(self.configs, 'relook_query_router_only', False)
                else output['role_router_probability']
            ),
            targets,
            reduction='none',
        )
        if informative.any():
            informative_weights = informative.to(per_query_loss.dtype)
            if getattr(self.configs, 'relook_query_router_only', False):
                # Train RR gains are substantially more positive than held-out
                # gains.  Equal class mass prevents the query residual from
                # degenerating into a second global relation bias.
                positive = informative & targets.bool()
                negative = informative & ~targets.bool()
                if positive.any() and negative.any():
                    informative_count = informative_weights.sum()
                    class_weights = torch.zeros_like(per_query_loss)
                    class_weights[positive] = (
                        0.5 * informative_count / positive.sum()
                    )
                    class_weights[negative] = (
                        0.5 * informative_count / negative.sum()
                    )
                    informative_weights = class_weights
            loss = (
                per_query_loss * informative_weights
            ).sum() / informative_weights.sum()
            target_rate = targets[informative].mean()
        else:
            # Preserve a differentiable zero for the first zero-adapter batch.
            loss = output['role_router_probability'].sum() * 0.0
            target_rate = targets.new_zeros(())
        return loss, target_rate, informative.to(per_query_loss.dtype).mean()

    def forward(self, ent_rel, src_ids, src_mask, target_ids=None):
        backbone = self._base_forward(
            ent_rel, src_ids, src_mask
        )
        base_logits = backbone['base_logits']
        prediction = backbone['prediction']
        structure_ent = backbone['structure_ent']
        structure_rel = backbone['structure_rel']
        source_has_image = self.has_image_mask[ent_rel[:, 0]]
        gate_mode = getattr(self.configs, 'relook_gate_mode', 'legacy')
        routing_candidate_ids = self._candidate_ids(base_logits)
        routing_visual_scores = None
        if gate_mode == 'utility':
            inspection_probability, uncertainty_stats = self.utility_gate(
                structure_ent,
                structure_rel,
                ent_rel[:, 1],
                base_logits,
                self.configs.relook_stat_topk,
            )
        elif gate_mode == 'evidence':
            routing_visual_scores = self._visual_scores(
                ent_rel, routing_candidate_ids
            )
            uncertainty_stats = self._evidence_statistics(
                base_logits, routing_candidate_ids, routing_visual_scores,
                ent_rel[:, 1],
            )
            inspection_probability = self.evidence_utility_gate(
                structure_ent, structure_rel, ent_rel[:, 1], uncertainty_stats
            )
        else:
            uncertainty, uncertainty_stats = self.uncertainty_gate(
                structure_ent,
                structure_rel,
                base_logits,
                self.configs.relook_stat_topk,
            )
            visuality = torch.sigmoid(self.relation_visuality(ent_rel[:, 1])).squeeze(-1)
            inspection_probability = uncertainty * visuality
        inspection_probability = (
            inspection_probability * source_has_image.to(inspection_probability.dtype)
        )
        visual_prior_probability = inspection_probability
        evidence_router_gain_logit = inspection_probability.new_zeros(
            inspection_probability.shape
        )
        evidence_router_effective_gain_logit = evidence_router_gain_logit
        if getattr(self.configs, 'relook_joint_evidence_router', False):
            (
                inspection_probability,
                evidence_router_gain_logit,
            ) = self.joint_evidence_router(
                structure_ent,
                structure_rel,
                visual_prior_probability,
                backbone['role_router_probability'],
                backbone['role_router_prior_probability'],
                backbone['role_router_query_gain_logit'],
                backbone['role_router_entropy'],
                backbone['role_router_margin'],
                backbone['role_router_delta_norm'],
                return_details=True,
            )
            (
                inspection_probability,
                evidence_router_effective_gain_logit,
            ) = self._role_aware_evidence_trust_region(
                visual_prior_probability,
                evidence_router_gain_logit,
                ent_rel[:, 1],
            )
            inspection_probability = (
                inspection_probability
                * source_has_image.to(inspection_probability.dtype)
            )
        candidate_ids = routing_candidate_ids
        natural_candidate_hits = None
        if target_ids is not None:
            natural_candidate_hits = candidate_ids.eq(target_ids.unsqueeze(1)).any(dim=1)
            if not natural_candidate_hits.all():
                candidate_ids = candidate_ids.clone()
                candidate_ids[~natural_candidate_hits, -1] = target_ids[~natural_candidate_hits]
        final_logits = base_logits
        full_visual_logits = None
        visual_scores = None
        max_blend = getattr(self.configs, 'relook_max_blend', 1.0)
        blend_cap = self._blend_cap(
            ent_rel[:, 1], inspection_probability
        ).to(base_logits.dtype)
        scope = getattr(self.configs, 'relook_scope', 'candidate')

        if self.training:
            inspect_mask = torch.ones_like(inspection_probability, dtype=torch.bool)
            if routing_visual_scores is not None and (
                natural_candidate_hits is None or natural_candidate_hits.all()
            ):
                visual_scores = routing_visual_scores
            else:
                visual_scores = self._visual_scores(ent_rel, candidate_ids)
            # ``F.normalize`` may retain FP32 for numerical stability under
            # autocast.  Scatter requires its source and destination to have
            # identical dtypes, so align evidence scores with Backbone logits.
            visual_scores = visual_scores.to(base_logits.dtype)
            blend = self.configs.relook_min_gate + (
                blend_cap - self.configs.relook_min_gate
            ) * inspection_probability
            blend = blend.to(base_logits.dtype)
            if scope in ('global', 'hybrid'):
                all_visual_scores = self.visual_verifier.score_all(
                    self._visual_entities(ent_rel[:, 0]),
                    self.visual_rel_embed(ent_rel[:, 1]),
                    self._visual_entities(),
                ).to(base_logits.dtype)
                if scope == 'global':
                    final_logits = base_logits + blend.unsqueeze(1) * all_visual_scores
                    full_visual_logits = base_logits + blend_cap.unsqueeze(1) * all_visual_scores
                else:
                    global_cap = self._global_blend_cap(
                        ent_rel[:, 1], inspection_probability
                    ).to(base_logits.dtype)
                    global_blend = global_cap * inspection_probability
                    global_blend = global_blend.to(base_logits.dtype)
                    final_logits = base_logits + global_blend.unsqueeze(1) * all_visual_scores
                    full_visual_logits = base_logits + global_cap.unsqueeze(1) * all_visual_scores
                    selected_base = base_logits.gather(1, candidate_ids)
                    final_logits = final_logits.scatter(
                        1, candidate_ids,
                        selected_base + blend.unsqueeze(1) * visual_scores,
                    )
                    full_visual_logits = full_visual_logits.scatter(
                        1, candidate_ids,
                        selected_base + blend_cap.unsqueeze(1) * visual_scores,
                    )
            else:
                selected_base = base_logits.gather(1, candidate_ids)
                reranked_scores = selected_base + blend.unsqueeze(1) * visual_scores
                final_logits = base_logits.scatter(1, candidate_ids, reranked_scores)
                full_visual_logits = base_logits.scatter(
                    1, candidate_ids,
                    selected_base + blend_cap.unsqueeze(1) * visual_scores
                )
                if scope == 'directional':
                    head_mask = ent_rel[:, 1] >= self.configs.n_rel
                    if head_mask.any():
                        head_visual_scores = self.visual_verifier.score_all(
                            self._visual_entities(ent_rel[head_mask, 0]),
                            self.visual_rel_embed(ent_rel[head_mask, 1]),
                            self._visual_entities(),
                        ).to(base_logits.dtype)
                        final_logits = final_logits.clone()
                        full_visual_logits = full_visual_logits.clone()
                        final_logits[head_mask] = (
                            base_logits[head_mask]
                            + blend[head_mask].unsqueeze(1) * head_visual_scores
                        )
                        full_visual_logits[head_mask] = (
                            base_logits[head_mask]
                            + blend_cap[head_mask].unsqueeze(1) * head_visual_scores
                        )
        else:
            inspect_mask = self._selection_mask(
                inspection_probability, base_logits, ent_rel, source_has_image
            )
            inspect_mask = inspect_mask & source_has_image
            if self.bypass_relation_ids:
                bypass_mask = torch.zeros_like(inspect_mask)
                for relation_id in self.bypass_relation_ids:
                    bypass_mask |= ent_rel[:, 1].eq(relation_id)
                inspect_mask = inspect_mask & ~bypass_mask
            if inspect_mask.any():
                active_blend = self._inference_blend(
                    blend_cap[inspect_mask], inspection_probability[inspect_mask]
                ).to(base_logits.dtype)
                if scope in ('global', 'hybrid'):
                    inspected_scores = self.visual_verifier.score_all(
                        self._visual_entities(ent_rel[inspect_mask, 0]),
                        self.visual_rel_embed(ent_rel[inspect_mask, 1]),
                        self._visual_entities(),
                    ).to(base_logits.dtype)
                    if scope == 'global':
                        inspected_logits = (
                            base_logits[inspect_mask]
                            + active_blend.unsqueeze(1) * inspected_scores
                        )
                    else:
                        global_cap = self._global_blend_cap(
                            ent_rel[inspect_mask, 1], inspection_probability[inspect_mask]
                        ).to(base_logits.dtype)
                        global_blend = self._inference_blend(
                            global_cap, inspection_probability[inspect_mask]
                        ).to(base_logits.dtype)
                        inspected_logits = (
                            base_logits[inspect_mask]
                            + global_blend.unsqueeze(1) * inspected_scores
                        )
                        inspected_ids = candidate_ids[inspect_mask]
                        selected_base = base_logits[inspect_mask].gather(1, inspected_ids)
                        candidate_scores = inspected_scores.gather(1, inspected_ids)
                        inspected_logits = inspected_logits.scatter(
                            1, inspected_ids,
                            selected_base + active_blend.unsqueeze(1) * candidate_scores,
                        )
                else:
                    inspected_ids = candidate_ids[inspect_mask]
                    inspected_scores = self._visual_scores(
                        ent_rel[inspect_mask], inspected_ids
                    ).to(base_logits.dtype)
                    selected_base = base_logits[inspect_mask].gather(1, inspected_ids)
                    reranked_scores = selected_base + (
                        active_blend.unsqueeze(1) * inspected_scores
                    )
                    inspected_logits = base_logits[inspect_mask].scatter(
                        1, inspected_ids, reranked_scores
                    )
                final_logits = base_logits.clone()
                final_logits[inspect_mask] = inspected_logits
                if scope == 'directional':
                    head_mask = inspect_mask & (ent_rel[:, 1] >= self.configs.n_rel)
                    if head_mask.any():
                        head_scores = self.visual_verifier.score_all(
                            self._visual_entities(ent_rel[head_mask, 0]),
                            self.visual_rel_embed(ent_rel[head_mask, 1]),
                            self._visual_entities(),
                        ).to(base_logits.dtype)
                        head_blend = self._inference_blend(
                            blend_cap[head_mask], inspection_probability[head_mask]
                        )
                        final_logits[head_mask] = (
                            base_logits[head_mask]
                            + head_blend.unsqueeze(1) * head_scores
                        )

        return {
            'logits': final_logits,
            'base_logits': base_logits,
            'shared_logits': backbone['shared_logits'],
            'adapter_logits': backbone['adapter_logits'],
            'prediction': prediction,
            'shared_prediction': backbone['shared_prediction'],
            'role_router_probability': backbone['role_router_probability'],
            'role_router_prior_probability': backbone[
                'role_router_prior_probability'
            ],
            'role_router_query_gain_logit': backbone[
                'role_router_query_gain_logit'
            ],
            'role_router_query_probability': backbone[
                'role_router_query_probability'
            ],
            'role_router_entropy': backbone['role_router_entropy'],
            'role_router_margin': backbone['role_router_margin'],
            'role_router_delta_norm': backbone['role_router_delta_norm'],
            'role_adapter_delta': backbone['role_adapter_delta'],
            'candidate_ids': candidate_ids,
            'visual_scores': visual_scores,
            'full_visual_logits': full_visual_logits,
            'natural_candidate_hits': natural_candidate_hits,
            'inspection_probability': inspection_probability,
            'visual_prior_probability': visual_prior_probability,
            'evidence_router_gain_logit': evidence_router_gain_logit,
            'evidence_router_effective_gain_logit': (
                evidence_router_effective_gain_logit
            ),
            'inspection_decision': inspect_mask,
            'uncertainty_stats': uncertainty_stats,
        }

    def _selection_mask(self, probability, logits, ent_rel, eligible):
        """Target-free policies; fixed-budget comparisons use equal batch quotas."""
        policy = getattr(self.configs, 'selection_policy', 'threshold')
        if policy == 'threshold':
            return (probability >= self.configs.relook_gate_threshold) & eligible
        if policy in ('none', 'always'):
            return torch.zeros_like(eligible) if policy == 'none' else eligible.clone()
        if policy == 'learned':
            score = probability.float()
        elif policy in ('entropy', 'margin'):
            values = logits.detach().float().topk(min(20, logits.size(1)), dim=1).values
            if policy == 'margin':
                score = -(values[:, 0] - values[:, 1])
            else:
                logp = values.log_softmax(dim=1)
                score = -(logp.exp() * logp).sum(dim=1)
        elif policy == 'random':
            # Integer mixing gives a repeatable query hash, independent of label,
            # batch order and CUDA RNG state. Duplicate queries share a priority;
            # the stable per-batch quota resolves ties at the budget boundary.
            key = ent_rel[:, 0].long() * (self.configs.n_rel * 2) + ent_rel[:, 1]
            key = (key + int(self.configs.seed)) & 0x7fffffff
            key = ((key ^ (key >> 16)) * 0x45d9f3b) & 0x7fffffff
            key = ((key ^ (key >> 16)) * 0x45d9f3b) & 0x7fffffff
            score = (key ^ (key >> 16)).double()
        else:
            raise ValueError(policy)
        count = min(int(eligible.sum()), round(len(probability) * self.configs.selection_budget))
        mask = torch.zeros_like(eligible)
        if count:
            order = torch.argsort(score.masked_fill(~eligible, -torch.inf), descending=True, stable=True)
            mask[order[:count]] = True
        return mask

    def training_step(self, batched_data, batch_idx):
        if self.configs.alpha_step > 0 and self.alpha < self.configs.alpha:
            self.alpha = min(self.alpha + self.configs.alpha_step, self.configs.alpha)
        if getattr(self.configs, 'relook_backbone_only', False):
            logits, prediction, _, _, _ = self._shared_backbone_forward(
                batched_data['ent_rel'], batched_data['source_ids'],
                batched_data['source_mask'],
            )
            labels = batched_data['labels']
            weight = self.configs.relook_truth_set_weight
            loss = (1.0 - weight) * self.loss_fn(logits, labels)
            if weight:
                loss = loss + weight * self._truth_set_loss(
                    logits, batched_data['ent_rel'], labels
                )
            if self.configs.n_lar > 0:
                loss = loss + self.alpha * self._language_adversarial_loss(
                    prediction, labels, batched_data['lars']
                )
            self.history['loss'].append(loss.detach().item())
            return {'loss': loss}

        labels = batched_data['labels']
        output = self(
            batched_data['ent_rel'],
            batched_data['source_ids'],
            batched_data['source_mask'],
            target_ids=labels,
        )
        base_loss = self.loss_fn(output['base_logits'], labels)
        rerank_loss = self.loss_fn(output['logits'], labels)
        role_adapter_loss = output['base_logits'].new_zeros(())
        role_router_loss = output['base_logits'].new_zeros(())
        role_router_target_rate = output['base_logits'].new_zeros(())
        role_router_informative_rate = output['base_logits'].new_zeros(())
        adapter_delta_norm = output['base_logits'].new_zeros(())
        role_adapters_enabled = getattr(
            self.configs, 'relook_role_adapters', False
        )
        learned_role_router = (
            role_adapters_enabled
            and getattr(
                self.configs, 'relook_role_router_mode', 'hybrid'
            ) != 'full'
        )
        rank_router_supervision = (
            learned_role_router
            and getattr(
                self.configs, 'relook_role_router_target', 'rank'
            ) == 'rank'
        )
        if role_adapters_enabled:
            role_adapter_loss = self.loss_fn(output['adapter_logits'], labels)
            adapter_delta_norm = output['role_adapter_delta'].pow(2).mean()
        if rank_router_supervision:
            (
                role_router_loss,
                role_router_target_rate,
                role_router_informative_rate,
            ) = self._rank_utility_router_loss(
                output, batched_data['ent_rel'], labels
            )
        elif learned_role_router:
            with torch.no_grad():
                shared_nll = F.cross_entropy(
                    output['shared_logits'], labels, reduction='none',
                    label_smoothing=self.configs.label_smoothing,
                )
                adapter_nll = F.cross_entropy(
                    output['adapter_logits'], labels, reduction='none',
                    label_smoothing=self.configs.label_smoothing,
                )
                router_temperature = max(
                    getattr(self.configs, 'relook_role_router_temperature', 0.5),
                    1e-4,
                )
                router_margin = getattr(
                    self.configs, 'relook_role_router_margin', 0.0
                )
                role_router_target = torch.sigmoid(
                    (shared_nll - adapter_nll - router_margin)
                    / router_temperature
                )
            role_router_loss = self._amp_safe_binary_cross_entropy(
                output['role_router_probability'], role_router_target
            )
            role_router_target_rate = role_router_target.mean()
        truth_set_weight = getattr(
            self.configs, 'relook_truth_set_weight', 0.0
        )
        truth_set_loss = output['base_logits'].new_zeros(())
        if truth_set_weight > 0:
            truth_set_base_loss = self._truth_set_loss(
                output['base_logits'], batched_data['ent_rel'], labels
            )
            truth_set_rerank_loss = self._truth_set_loss(
                output['logits'], batched_data['ent_rel'], labels
            )
            base_loss = (
                (1.0 - truth_set_weight) * base_loss
                + truth_set_weight * truth_set_base_loss
            )
            rerank_loss = (
                (1.0 - truth_set_weight) * rerank_loss
                + truth_set_weight * truth_set_rerank_loss
            )
            if role_adapters_enabled:
                truth_set_adapter_loss = self._truth_set_loss(
                    output['adapter_logits'], batched_data['ent_rel'], labels
                )
                role_adapter_loss = (
                    (1.0 - truth_set_weight) * role_adapter_loss
                    + truth_set_weight * truth_set_adapter_loss
                )
            if learned_role_router and not rank_router_supervision:
                # The router learns whether the role expert improves the whole
                # known answer set, instead of overfitting to one sampled answer.
                with torch.no_grad():
                    shared_truth_nll = self._truth_set_loss(
                        output['shared_logits'], batched_data['ent_rel'], labels,
                        reduction='none',
                    )
                    adapter_truth_nll = self._truth_set_loss(
                        output['adapter_logits'], batched_data['ent_rel'], labels,
                        reduction='none',
                    )
                    router_temperature = max(
                        getattr(
                            self.configs,
                            'relook_role_router_temperature',
                            0.5,
                        ),
                        1e-4,
                    )
                    router_margin = getattr(
                        self.configs, 'relook_role_router_margin', 0.0
                    )
                    role_router_target = torch.sigmoid(
                        (shared_truth_nll - adapter_truth_nll - router_margin)
                        / router_temperature
                    )
                role_router_loss = self._amp_safe_binary_cross_entropy(
                    output['role_router_probability'], role_router_target
                )
                role_router_target_rate = role_router_target.mean()
            truth_set_loss = 0.5 * (
                truth_set_base_loss + truth_set_rerank_loss
            )

        target_positions = output['candidate_ids'].eq(labels.unsqueeze(1)).long().argmax(dim=1)
        visual_loss = F.cross_entropy(output['visual_scores'], target_positions)
        budget_target = getattr(self.configs, 'relook_target_rate', 0.5)
        budget_loss = F.relu(
            output['inspection_probability'].mean() - budget_target
        ).square()
        gate_loss = output['inspection_probability'].new_zeros(())
        utility_rate = output['inspection_probability'].new_zeros(())
        raw_utility_rate = output['inspection_probability'].new_zeros(())
        filtered_utility_rate = output['inspection_probability'].new_zeros(())
        utility_target_disagreement = output[
            'inspection_probability'
        ].new_zeros(())
        corrected_false_negative_rate = output[
            'inspection_probability'
        ].new_zeros(())
        if getattr(
            self.configs, 'relook_gate_mode', 'legacy'
        ) in ('utility', 'evidence'):
            with torch.no_grad():
                raw_base_rr = self._raw_reciprocal_rank(
                    output['base_logits'], labels
                )
                raw_visual_rr = self._raw_reciprocal_rank(
                    output['full_visual_logits'], labels
                )
                filtered_base_rr = self._training_filtered_reciprocal_rank(
                    output['base_logits'], batched_data['ent_rel'], labels
                )
                filtered_visual_rr = self._training_filtered_reciprocal_rank(
                    output['full_visual_logits'], batched_data['ent_rel'], labels
                )
                raw_utility_gain = raw_visual_rr - raw_base_rr
                filtered_utility_gain = filtered_visual_rr - filtered_base_rr
                utility_margin = getattr(
                    self.configs, 'relook_utility_margin', 0.0
                )
                raw_utility_target = raw_utility_gain.gt(utility_margin)
                filtered_utility_target = filtered_utility_gain.gt(
                    utility_margin
                )
                target_mode = getattr(
                    self.configs,
                    'relook_visual_utility_target',
                    'filtered',
                )
                if target_mode == 'filtered':
                    utility_gain = filtered_utility_gain
                    utility_target = filtered_utility_target.float()
                elif target_mode == 'raw':
                    utility_gain = raw_utility_gain
                    utility_target = raw_utility_target.float()
                else:
                    raise ValueError(
                        f'Unsupported visual utility target: {target_mode}'
                    )
                if getattr(self.configs, 'relook_scope', 'candidate') != 'global':
                    utility_target = utility_target * output['natural_candidate_hits'].float()
                raw_utility_rate = raw_utility_target.float().mean()
                filtered_utility_rate = filtered_utility_target.float().mean()
                utility_target_disagreement = raw_utility_target.ne(
                    filtered_utility_target
                ).float().mean()
                corrected_false_negative_rate = (
                    filtered_utility_target & ~raw_utility_target
                ).float().mean()
                utility_weight = 1.0 + getattr(
                    self.configs, 'relook_utility_gain_scale', 4.0
                ) * utility_gain.abs().clamp_max(1.0)
                positive_rate = utility_target.mean().clamp(1e-3, 1.0 - 1e-3)
                class_weight = torch.where(
                    utility_target.bool(),
                    0.5 / positive_rate,
                    0.5 / (1.0 - positive_rate),
                )
            gate_loss = self._amp_safe_binary_cross_entropy(
                output['inspection_probability'], utility_target,
                weight=utility_weight * class_weight,
            )
            utility_rate = utility_target.mean()
        loss = (
            self.configs.relook_base_weight * base_loss
            + self.configs.relook_rerank_weight * rerank_loss
            + self.configs.relook_visual_weight * visual_loss
            + self.configs.relook_budget_weight * budget_loss
            + getattr(self.configs, 'relook_gate_weight', 0.0) * gate_loss
            + getattr(self.configs, 'relook_role_adapter_weight', 0.0)
            * role_adapter_loss
            + getattr(self.configs, 'relook_role_router_weight', 0.0)
            * role_router_loss
            + getattr(self.configs, 'relook_adapter_norm_weight', 0.0)
            * adapter_delta_norm
        )

        if self.configs.n_lar > 0:
            lar_loss = self._language_adversarial_loss(
                output['prediction'], labels, batched_data['lars'],
            )
            loss = loss + self.alpha * lar_loss

        self.history['loss'].append(loss.detach().item())
        self.log('train_relook_gate', budget_loss, prog_bar=True)
        self.log('train_gate_loss', gate_loss, prog_bar=True)
        self.log('train_utility_rate', utility_rate, prog_bar=True)
        self.log('train_raw_utility_rate', raw_utility_rate)
        self.log('train_filtered_utility_rate', filtered_utility_rate)
        self.log(
            'train_utility_target_disagreement', utility_target_disagreement
        )
        self.log(
            'train_corrected_false_negative_rate',
            corrected_false_negative_rate,
        )
        self.log('train_truth_set_loss', truth_set_loss, prog_bar=True)
        self.log('train_role_adapter_loss', role_adapter_loss, prog_bar=True)
        self.log('train_role_router_loss', role_router_loss, prog_bar=True)
        self.log('train_role_router_target', role_router_target_rate, prog_bar=True)
        self.log(
            'train_role_router_informative',
            role_router_informative_rate,
            prog_bar=True,
        )
        self.log(
            'train_role_router_probability',
            output['role_router_probability'].mean(),
            prog_bar=True,
        )
        self.log(
            'train_role_router_prior',
            output['role_router_prior_probability'].mean(),
        )
        self.log(
            'train_role_router_query_gain',
            output['role_router_query_gain_logit'].mean(),
        )
        self.log(
            'train_role_router_entropy', output['role_router_entropy'].mean()
        )
        self.log(
            'train_role_router_margin', output['role_router_margin'].mean()
        )
        self.log(
            'train_role_router_delta_norm',
            output['role_router_delta_norm'].mean(),
        )
        self.log(
            'train_evidence_router_gain',
            output['evidence_router_gain_logit'].mean(),
        )
        self.log('train_adapter_delta_norm', adapter_delta_norm, prog_bar=False)
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        output = self(
            batched_data['ent_rel'],
            batched_data['source_ids'],
            batched_data['source_mask'],
        )
        self.inspection_probabilities.append(
            output['inspection_probability'].detach().cpu()
        )
        self.inspection_decisions.append(output['inspection_decision'].detach().cpu())
        logits = output['logits'].detach()
        base_logits = output['base_logits'].detach()
        shared_logits = output['shared_logits'].detach()
        adapter_logits = output['adapter_logits'].detach()
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        target_ent = batched_data['tgt_ent']
        target_tensor = torch.as_tensor(target_ent, device=ent_rel.device)
        triples = batched_data['triple']
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        candidate_hits = output['candidate_ids'].eq(target_tensor.unsqueeze(1)).any(dim=1)
        final_ranks, final_top1 = self._filtered_ranks(
            logits.clone(), src_ent, rel, target_ent, triples, gt, batch_idx
        )
        base_ranks, base_top1 = self._filtered_ranks(
            base_logits.clone(), src_ent, rel, target_ent, triples, gt, batch_idx
        )
        shared_ranks, shared_top1 = self._filtered_ranks(
            shared_logits.clone(), src_ent, rel, target_ent, triples, gt, batch_idx
        )
        adapter_ranks, adapter_top1 = self._filtered_ranks(
            adapter_logits.clone(), src_ent, rel, target_ent, triples, gt, batch_idx
        )
        answer_counts = []
        for index in range(len(src_ent)):
            query = (src_ent[index].item(), rel[index].item())
            if self.configs.is_temporal:
                query = query + (triples[index][3],)
            answer_counts.append(len(gt[query]))
        return {
            'final_ranks': final_ranks,
            'base_ranks': base_ranks,
            'shared_ranks': shared_ranks,
            'adapter_ranks': adapter_ranks,
            'candidate_hits': candidate_hits.detach().cpu().numpy(),
            'final_top1': final_top1,
            'base_top1': base_top1,
            'shared_top1': shared_top1,
            'adapter_top1': adapter_top1,
            'source_ids': src_ent.detach().cpu().numpy(),
            'target_ids': np.asarray(target_ent, dtype=np.int64),
            'relation_ids': rel.detach().cpu().numpy(),
            'answer_counts': np.asarray(answer_counts, dtype=np.int64),
            'inspection_probability': output['inspection_probability'].detach().cpu().numpy(),
            'visual_prior_probability': output[
                'visual_prior_probability'
            ].detach().cpu().numpy(),
            'evidence_router_gain_logit': output[
                'evidence_router_gain_logit'
            ].detach().cpu().numpy(),
            'evidence_router_effective_gain_logit': output[
                'evidence_router_effective_gain_logit'
            ].detach().cpu().numpy(),
            'inspection_decision': output['inspection_decision'].detach().cpu().numpy(),
            'role_router_probability': output['role_router_probability'].detach().cpu().numpy(),
            'role_router_prior_probability': output[
                'role_router_prior_probability'
            ].detach().cpu().numpy(),
            'role_router_query_gain_logit': output[
                'role_router_query_gain_logit'
            ].detach().cpu().numpy(),
            'role_router_entropy': output[
                'role_router_entropy'
            ].detach().cpu().numpy(),
            'role_router_margin': output[
                'role_router_margin'
            ].detach().cpu().numpy(),
            'role_router_delta_norm': output[
                'role_router_delta_norm'
            ].detach().cpu().numpy(),
        }

    def _filtered_ranks(self, logits, src_ent, rel, target_ent, triples, gt, batch_idx):
        for index in range(len(src_ent)):
            head = src_ent[index].item()
            tail = target_ent[index]
            relation = rel[index].item()
            if self.configs.is_temporal:
                target_filter = gt[(head, relation, triples[index][3])]
            else:
                target_filter = gt[(head, relation)]
            target_score = logits[index, tail].item()
            logits[index, target_filter] = self._MASKING_VALUE
            logits[index, tail] = target_score

        ranking = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        ranks = [
            np.where(ranking[index] == target_ent[index])[0][0] + 1
            for index in range(len(target_ent))
        ]
        if self.configs.use_log_ranks:
            filename = os.path.join(
                self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp'
            )
            self._log_ranks(filename, triples, ranking, ranks, batch_idx)
        top1 = ranking[:, 0].astype(np.int64)
        return np.asarray(ranks, dtype=np.int64), top1

    def validation_epoch_end(self, outputs):
        tail_ranks = np.concatenate([item['final_ranks'] for item in outputs[0]])
        head_ranks = np.concatenate([item['final_ranks'] for item in outputs[1]])
        base_tail_ranks = np.concatenate([item['base_ranks'] for item in outputs[0]])
        base_head_ranks = np.concatenate([item['base_ranks'] for item in outputs[1]])
        shared_tail_ranks = np.concatenate([
            item['shared_ranks'] for item in outputs[0]
        ])
        shared_head_ranks = np.concatenate([
            item['shared_ranks'] for item in outputs[1]
        ])
        adapter_tail_ranks = np.concatenate([
            item['adapter_ranks'] for item in outputs[0]
        ])
        adapter_head_ranks = np.concatenate([
            item['adapter_ranks'] for item in outputs[1]
        ])
        candidate_hits = np.concatenate([
            item['candidate_hits'] for loader_outputs in outputs for item in loader_outputs
        ])
        performance = get_performance(self, tail_ranks, head_ranks)
        self.history['perf'] = performance
        base_ranks = np.concatenate([base_tail_ranks, base_head_ranks])
        shared_ranks = np.concatenate([shared_tail_ranks, shared_head_ranks])
        adapter_ranks = np.concatenate([adapter_tail_ranks, adapter_head_ranks])
        final_ranks = np.concatenate([tail_ranks, head_ranks])
        base_mrr = np.mean(1.0 / base_ranks)
        shared_mrr = np.mean(1.0 / shared_ranks)
        adapter_mrr = np.mean(1.0 / adapter_ranks)
        final_mrr = np.mean(1.0 / final_ranks)
        probabilities = torch.cat(self.inspection_probabilities)
        decisions = torch.cat(self.inspection_decisions).float()
        invocation_rate = decisions.mean().item()
        self.log('val_base_mrr', float(base_mrr))
        self.log('val_shared_mrr', float(shared_mrr))
        self.log('val_adapter_mrr', float(adapter_mrr))
        self.log('val_role_mrr_gain', float(base_mrr - shared_mrr))
        self.log('val_mrr_gain', float(final_mrr - base_mrr))
        self.log('val_inspection_rate', invocation_rate)
        print('Epoch:', self.current_epoch)
        print(performance)
        print(
            'ReLook diagnostics: base_mrr=%.6f, delta_mrr=%+.6f, '
            'top%d_recall=%.2f%%, gate=%.4f, invocation_rate=%.2f%%'
            % (
                base_mrr,
                final_mrr - base_mrr,
                self.configs.relook_topk,
                candidate_hits.mean() * 100,
                probabilities.mean().item(),
                invocation_rate * 100,
            )
        )
        role_probabilities = np.concatenate([
            item['role_router_probability']
            for loader_outputs in outputs for item in loader_outputs
        ])
        role_prior_probabilities = np.concatenate([
            item['role_router_prior_probability']
            for loader_outputs in outputs for item in loader_outputs
        ])
        role_query_gains = np.concatenate([
            item['role_router_query_gain_logit']
            for loader_outputs in outputs for item in loader_outputs
        ])
        tail_count = len(tail_ranks)
        print(
            'Role-Adapter diagnostics: shared_mrr=%.6f, routed_mrr=%.6f, '
            'full_adapter_mrr=%.6f, role_gain=%+.6f, '
            'tail_route=%.4f, head_route=%.4f, prior=%.4f, query_gain=%+.4f'
            % (
                shared_mrr,
                base_mrr,
                adapter_mrr,
                base_mrr - shared_mrr,
                role_probabilities[:tail_count].mean(),
                role_probabilities[tail_count:].mean(),
                role_prior_probabilities.mean(),
                role_query_gains.mean(),
            )
        )
        if self.configs.badcase_report:
            self._export_badcase_report(outputs)

    def on_validation_epoch_start(self):
        self.inspection_probabilities = []
        self.inspection_decisions = []

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        self.inspection_probabilities = []
        self.inspection_decisions = []

    def configure_optimizers(self):
        adapter_parameters = []
        router_parameters = []
        backbone_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith((
                'tail_role_adapter.', 'head_role_adapter.',
                'shared_deep_query_adapter.',
                'shared_deep_prompt_expert.',
                'tail_deep_prompt_expert.',
                'head_deep_prompt_expert.',
            )):
                adapter_parameters.append(parameter)
            elif name.startswith((
                'relation_role_router.', 'joint_evidence_router.',
            )):
                router_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)

        parameter_groups = []
        if backbone_parameters:
            parameter_groups.append({
                'params': backbone_parameters,
                'lr': self.configs.lr,
            })
        if adapter_parameters:
            adapter_lr = getattr(self.configs, 'relook_adapter_lr', -1.0)
            parameter_groups.append({
                'params': adapter_parameters,
                'lr': self.configs.lr if adapter_lr <= 0 else adapter_lr,
            })
        if router_parameters:
            router_lr = getattr(self.configs, 'relook_role_router_lr', -1.0)
            parameter_groups.append({
                'params': router_parameters,
                'lr': self.configs.lr if router_lr <= 0 else router_lr,
            })
        return torch.optim.AdamW(parameter_groups)

    @staticmethod
    def _concat_output(outputs, key):
        return np.concatenate([
            item[key] for loader_outputs in outputs for item in loader_outputs
        ])

    def _export_badcase_report(self, outputs):
        report_dir = Path(self.configs.badcase_report)
        report_dir.mkdir(parents=True, exist_ok=True)
        fields = {
            key: self._concat_output(outputs, key)
            for key in (
                'final_ranks', 'base_ranks', 'shared_ranks', 'adapter_ranks',
                'candidate_hits', 'final_top1', 'base_top1', 'shared_top1',
                'adapter_top1', 'source_ids', 'target_ids', 'relation_ids',
                'answer_counts', 'inspection_probability',
                'visual_prior_probability', 'evidence_router_gain_logit',
                'evidence_router_effective_gain_logit',
                'inspection_decision',
                'role_router_probability', 'role_router_prior_probability',
                'role_router_query_gain_logit', 'role_router_entropy',
                'role_router_margin', 'role_router_delta_norm',
            )
        }
        count = len(fields['final_ranks'])
        directions = np.concatenate([
            np.full(sum(len(item['final_ranks']) for item in outputs[index]), index)
            for index in range(2)
        ])
        base_rr = 1.0 / fields['base_ranks']
        shared_rr = 1.0 / fields['shared_ranks']
        adapter_rr = 1.0 / fields['adapter_ranks']
        final_rr = 1.0 / fields['final_ranks']
        rr_gain = final_rr - base_rr
        adapter_utility = adapter_rr - shared_rr

        def binary_auc(labels, scores):
            """Tie-aware Mann-Whitney AUC without an sklearn dependency."""
            labels = np.asarray(labels, dtype=bool)
            scores = np.asarray(scores, dtype=float)
            positive_count = int(labels.sum())
            negative_count = int((~labels).sum())
            if positive_count == 0 or negative_count == 0:
                return None

            order = np.argsort(scores, kind='mergesort')
            sorted_scores = scores[order]
            ranks = np.empty(len(scores), dtype=float)
            start = 0
            while start < len(scores):
                end = start + 1
                while (
                    end < len(scores)
                    and sorted_scores[end] == sorted_scores[start]
                ):
                    end += 1
                # Average of the one-indexed ranks [start + 1, end].
                ranks[order[start:end]] = (start + end + 1) / 2.0
                start = end
            positive_rank_sum = ranks[labels].sum()
            return float(
                (
                    positive_rank_sum
                    - positive_count * (positive_count + 1) / 2.0
                )
                / (positive_count * negative_count)
            )

        def metrics(mask):
            selected = int(mask.sum())
            if selected == 0:
                return {'count': 0}
            return {
                'count': selected,
                'base_mrr': float(base_rr[mask].mean()),
                'final_mrr': float(final_rr[mask].mean()),
                'mrr_gain': float(rr_gain[mask].mean()),
                'hit10': float((fields['final_ranks'][mask] <= 10).mean()),
                'candidate_recall': float(fields['candidate_hits'][mask].mean()),
                'improved_rate': float((rr_gain[mask] > 0).mean()),
                'harmed_rate': float((rr_gain[mask] < 0).mean()),
            }

        all_mask = np.ones(count, dtype=bool)
        rank_buckets = {
            'rank_1': fields['base_ranks'] == 1,
            'rank_2_10': (fields['base_ranks'] >= 2) & (fields['base_ranks'] <= 10),
            'rank_11_128': (fields['base_ranks'] >= 11) & (fields['base_ranks'] <= 128),
            'rank_gt_128': fields['base_ranks'] > 128,
        }
        ambiguity_buckets = {
            'one_answer': fields['answer_counts'] == 1,
            'answers_2_5': (fields['answer_counts'] >= 2) & (fields['answer_counts'] <= 5),
            'answers_gt_5': fields['answer_counts'] > 5,
        }
        informative_router_mask = adapter_utility != 0
        helpful_router_mask = adapter_utility[informative_router_mask] > 0
        informative_router_probability = fields['role_router_probability'][
            informative_router_mask
        ]

        def optional_mean(values):
            values = np.asarray(values)
            return float(values.mean()) if values.size else None

        summary = {
            'overall': metrics(all_mask),
            'tail': metrics(directions == 0),
            'head': metrics(directions == 1),
            'candidate_hit': metrics(fields['candidate_hits'].astype(bool)),
            'candidate_miss': metrics(~fields['candidate_hits'].astype(bool)),
            'rank_buckets': {name: metrics(mask) for name, mask in rank_buckets.items()},
            'ambiguity_buckets': {
                name: metrics(mask) for name, mask in ambiguity_buckets.items()
            },
            'oracle_selector_mrr': float(np.maximum(base_rr, final_rr).mean()),
            'inspection_rate': float(fields['inspection_decision'].mean()),
            'mean_inspection_probability': float(fields['inspection_probability'].mean()),
            'evidence_router': {
                'mean_visual_prior_probability': float(
                    fields['visual_prior_probability'].mean()
                ),
                'mean_probability': float(
                    fields['inspection_probability'].mean()
                ),
                'mean_gain_logit': float(
                    fields['evidence_router_gain_logit'].mean()
                ),
                'mean_effective_gain_logit': float(
                    fields['evidence_router_effective_gain_logit'].mean()
                ),
            },
            'role_adapter': {
                'shared_mrr': float(shared_rr.mean()),
                'routed_mrr': float(base_rr.mean()),
                'full_adapter_mrr': float(adapter_rr.mean()),
                'routed_gain': float((base_rr - shared_rr).mean()),
                'tail_router_probability': float(
                    fields['role_router_probability'][directions == 0].mean()
                ),
                'head_router_probability': float(
                    fields['role_router_probability'][directions == 1].mean()
                ),
                'mean_prior_probability': float(
                    fields['role_router_prior_probability'].mean()
                ),
                'mean_query_gain_logit': float(
                    fields['role_router_query_gain_logit'].mean()
                ),
                'full_adapter_gain': float(adapter_utility.mean()),
                'informative_rate': float(informative_router_mask.mean()),
                'helpful_rate_among_informative': optional_mean(
                    helpful_router_mask
                ),
                'utility_auc': binary_auc(
                    helpful_router_mask,
                    informative_router_probability,
                ),
                'prior_utility_auc': binary_auc(
                    helpful_router_mask,
                    fields['role_router_prior_probability'][
                        informative_router_mask
                    ],
                ),
                'query_gain_utility_auc': binary_auc(
                    helpful_router_mask,
                    fields['role_router_query_gain_logit'][
                        informative_router_mask
                    ],
                ),
                'helpful_router_probability': optional_mean(
                    informative_router_probability[helpful_router_mask]
                ),
                'harmful_router_probability': optional_mean(
                    informative_router_probability[~helpful_router_mask]
                ),
            },
        }
        with open(report_dir / 'summary.json', 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        rows = []
        for index in range(count):
            relation_id = int(fields['relation_ids'][index])
            raw_relation_id = relation_id % self.configs.n_rel
            rows.append({
                'index': index,
                'direction': 'tail' if directions[index] == 0 else 'head',
                'source_id': int(fields['source_ids'][index]),
                'source_name': self.ent_names[int(fields['source_ids'][index])],
                'relation_id': raw_relation_id,
                'relation_name': self.rel_names[raw_relation_id],
                'relation_train_frequency': int(self.relation_frequency[raw_relation_id]),
                'target_id': int(fields['target_ids'][index]),
                'target_name': self.ent_names[int(fields['target_ids'][index])],
                'base_top1_id': int(fields['base_top1'][index]),
                'base_top1_name': self.ent_names[int(fields['base_top1'][index])],
                'shared_top1_id': int(fields['shared_top1'][index]),
                'shared_top1_name': self.ent_names[int(fields['shared_top1'][index])],
                'adapter_top1_id': int(fields['adapter_top1'][index]),
                'adapter_top1_name': self.ent_names[int(fields['adapter_top1'][index])],
                'final_top1_id': int(fields['final_top1'][index]),
                'final_top1_name': self.ent_names[int(fields['final_top1'][index])],
                'base_rank': int(fields['base_ranks'][index]),
                'shared_rank': int(fields['shared_ranks'][index]),
                'adapter_rank': int(fields['adapter_ranks'][index]),
                'final_rank': int(fields['final_ranks'][index]),
                'role_reciprocal_rank_gain': float(base_rr[index] - shared_rr[index]),
                'reciprocal_rank_gain': float(rr_gain[index]),
                'candidate_hit': bool(fields['candidate_hits'][index]),
                'answer_count': int(fields['answer_counts'][index]),
                'inspection_probability': float(fields['inspection_probability'][index]),
                'visual_prior_probability': float(
                    fields['visual_prior_probability'][index]
                ),
                'evidence_router_gain_logit': float(
                    fields['evidence_router_gain_logit'][index]
                ),
                'evidence_router_effective_gain_logit': float(
                    fields['evidence_router_effective_gain_logit'][index]
                ),
                'inspected': bool(fields['inspection_decision'][index]),
                'role_router_probability': float(
                    fields['role_router_probability'][index]
                ),
                'role_router_prior_probability': float(
                    fields['role_router_prior_probability'][index]
                ),
                'role_router_query_gain_logit': float(
                    fields['role_router_query_gain_logit'][index]
                ),
                'backbone_entropy': float(
                    fields['role_router_entropy'][index]
                ),
                'backbone_margin': float(
                    fields['role_router_margin'][index]
                ),
                'adapter_delta_norm': float(
                    fields['role_router_delta_norm'][index]
                ),
            })
        with open(report_dir / 'queries.csv', 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        relation_rows = []
        for relation_id in np.unique(fields['relation_ids']):
            mask = fields['relation_ids'] == relation_id
            raw_relation_id = int(relation_id) % self.configs.n_rel
            relation_rows.append({
                'relation_id': int(relation_id),
                'raw_relation_id': raw_relation_id,
                'direction': 'tail' if relation_id < self.configs.n_rel else 'head',
                'relation_name': self.rel_names[raw_relation_id],
                'train_frequency': int(self.relation_frequency[raw_relation_id]),
                'role_router_probability': float(
                    fields['role_router_probability'][mask].mean()
                ),
                'role_router_prior_probability': float(
                    fields['role_router_prior_probability'][mask].mean()
                ),
                'role_router_query_gain_logit': float(
                    fields['role_router_query_gain_logit'][mask].mean()
                ),
                'role_mrr_gain': float((base_rr[mask] - shared_rr[mask]).mean()),
                **metrics(mask),
            })
        relation_rows.sort(key=lambda row: row.get('mrr_gain', 0.0))
        with open(report_dir / 'relations.csv', 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=relation_rows[0].keys())
            writer.writeheader()
            writer.writerows(relation_rows)
        print(f'BadCase report written to {report_dir.resolve()}')

    def _log_ranks(self, filename, triples, ranking, ranks, batch_idx):
        with open(filename, 'a', encoding='utf-8') as file:
            for index, triple in enumerate(triples):
                if not self.configs.is_temporal:
                    head, tail, relation = triple
                    timestamp = ''
                else:
                    head, tail, relation, timestamp = triple
                    timestamp = ' | ' + timestamp
                triple_text = (
                    self.ent_names[head] + ' [' + self.ent_descs[head] + '] | '
                    + self.rel_names[relation] + ' | ' + self.ent_names[tail]
                    + ' [' + self.ent_descs[tail] + '] ' + timestamp
                )
                file.write(
                    f'{batch_idx * self.configs.val_batch_size + index}. '
                    f'{triple_text} => rank: {ranks[index]}\n'
                )
                for candidate_rank, entity in enumerate(ranking[index, :10], start=1):
                    marker = '*' if candidate_rank == ranks[index] else ' '
                    file.write(
                        f'\t{candidate_rank:2d}{marker} {self.ent_names[entity]} '
                        f'[{self.ent_descs[entity]}] ({entity})\n'
                    )
