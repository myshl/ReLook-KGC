import time
import torch
from pytorch_lightning.callbacks import Callback


class PrintingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        print('Epoch: %4d, ' % trainer.current_epoch, flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if len(pl_module.history['loss']) == 0:
            return
        loss = pl_module.history['loss']
        avg_loss = sum(loss) / len(loss)
        pl_module.history['loss'] = []
        print('Total time: %4ds, loss: %2.4f' % (int(time.time() - self.start), avg_loss), flush=True)
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(pl_module.device) / 2 ** 30
            peak_reserved = torch.cuda.max_memory_reserved(pl_module.device) / 2 ** 30
            print(
                'Peak CUDA memory: allocated=%2.2f GiB, reserved=%2.2f GiB'
                % (peak_allocated, peak_reserved),
                flush=True,
            )
        print('-' * 50, flush=True)

    def on_validation_start(self, trainer, pl_module):
        if hasattr(self, 'start'):
            print('Training time: %4ds' % (int(time.time() - self.start)), flush=True)
        self.val_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        print(pl_module.history['perf'], flush=True)
        print('Validation time: %4ds' % (int(time.time() - self.val_start)), flush=True)

    def on_test_end(self, trainer, pl_module):
        print(flush=True)
        print('=' * 50, flush=True)
        print('Epoch: test', flush=True)
        print(pl_module.history['perf'], flush=True)
        print('=' * 50, flush=True)
