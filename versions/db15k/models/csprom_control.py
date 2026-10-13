"""Official CSProm forward/objective with the suite's data and logging interface.

Source: chenchens190009/CSProm-KG@9a8072951beda3e5c609bd121c900b153466ca28.
The reference is preserved verbatim. The suite matches dimensions, initialization
and validation budget; it is not a reproduction of the paper's tuned scores.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from helper import get_performance
from models.prompter import Prompter

_path = Path(__file__).resolve().parents[1] / 'reference/csprom/P_model.py'
_spec = importlib.util.spec_from_file_location('official_csprom', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class CSPromControl(_module.KGCPromptTuner):
    def __init__(self, configs, text_dict, gt):
        super().__init__(configs, text_dict, gt)
        # The inherited local Prompter default is half-depth for historical
        # multimodal code. Restore the official full-depth constructor.
        self.prompter = Prompter(
            self.plm_configs, configs.embed_dim, configs.prompt_length,
            num_layers=self.plm_configs.num_hidden_layers,
        )
        import pickle
        feature_name = 'FB15K-237' if configs.dataset.lower() == 'fb15k-237' else configs.dataset
        for suffix, embedding in [('entity', self.ent_embed), ('relation', self.rel_embed)]:
            with open(Path(configs.feature_dir) / f'{feature_name}_{suffix}_embed.pkl', 'rb') as f:
                array = pickle.load(f)
            with torch.no_grad():
                embedding.weight.copy_(torch.as_tensor(array))
        # Match all common random weights, not just the structural table. The
        # canonical constructor is seeded in a fork so data-order RNG is kept.
        from models.relook_model import ReLookKG
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(configs.seed)
            canonical = ReLookKG(configs, text_dict, gt)
        for target, source in [(self.plm, canonical.plm),
                               (self.prompter, canonical.gat_prompter),
                               (self.fc, canonical.prompt_fc),
                               (self.graph_model, canonical.graph_model)]:
            target.load_state_dict(source.state_dict(), strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.plm.eval()  # same frozen-encoder dropout regime as the matched control
        return self

    def validation_epoch_end(self, outs):
        self.history['perf'] = get_performance(
            self, np.concatenate(outs[0]), np.concatenate(outs[1])
        )
        print(self.history['perf'], flush=True)
