"""Persist convergence and numeric audits without modifying the objective."""
import json
import time
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback


class RunAudit(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.start = time.monotonic()
        self.nonfinite_grad_batches = 0
        self.grad_batches = 0
        self.path = Path(pl_module.configs.save_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / 'config.json').open('w') as f:
            json.dump(vars(pl_module.configs), f, indent=2)
        with (self.path / 'parameters.json').open('w') as f:
            json.dump({
                'total_instantiated': sum(p.numel() for p in pl_module.parameters()),
                'trainable': sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
                'trainable_names': [n for n,p in pl_module.named_parameters() if p.requires_grad],
            }, f, indent=2)

    def on_before_backward(self, trainer, pl_module, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss; refusing to record a successful run')

    def on_after_backward(self, trainer, pl_module):
        self.grad_batches += 1
        # AMP may legitimately skip initial overflows, but repeated ones must
        # not masquerade as training with an unchanged visual checkpoint.
        checks = [torch.isfinite(p.grad).all() for p in pl_module.parameters() if p.grad is not None]
        finite = bool(torch.stack(checks).all()) if checks else False
        self.nonfinite_grad_batches += int(not finite)
        if self.grad_batches >= 30 and self.nonfinite_grad_batches / self.grad_batches > 0.25:
            raise FloatingPointError('More than 25% nonfinite-gradient batches')

    def on_validation_end(self, trainer, pl_module):
        if str(trainer.state.fn) not in ('fit', 'TrainerFn.FITTING') or not hasattr(self, 'path'):
            return
        row = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'elapsed_seconds': time.monotonic() - self.start,
            'nonfinite_grad_batches': self.nonfinite_grad_batches,
            'grad_batches': self.grad_batches,
            'metrics': {k: float(v) for k,v in trainer.callback_metrics.items() if v.numel() == 1},
        }
        with (self.path / 'convergence.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
