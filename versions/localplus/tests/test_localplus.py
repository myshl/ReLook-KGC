import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tuning_utils import similarity_margin_loss, complete_training_negatives, prefer_initial
from models.relook_model import ReLookKG


class LocalplusTest(unittest.TestCase):
    def test_margin_gradient_increases_positive_similarity(self):
        positive = torch.tensor([0.], requires_grad=True)
        negative = torch.tensor([0.], requires_grad=True)
        similarity_margin_loss(positive, negative, 1.).backward()
        self.assertLess(positive.grad.item(), 0)
        self.assertGreater(negative.grad.item(), 0)
        self.assertEqual(similarity_margin_loss(torch.tensor([2.]), torch.tensor([0.]), 1.).item(), 0.)

    def test_training_negative_fill_never_samples_known_answers(self):
        np.random.seed(41504)
        for _ in range(30):
            result = complete_training_negatives([1, 2, 3, 3], [1, 2, 5], 10, 6)
            self.assertEqual(len(result), 6)
            self.assertEqual(len(set(result)), 6)
            self.assertFalse(set(result) & {1, 2, 5})

    def test_initial_validation_checkpoint_is_eligible(self):
        self.assertTrue(prefer_initial(.36, .35))
        self.assertTrue(prefer_initial(.36, .36))
        self.assertFalse(prefer_initial(.36, .37))
        self.assertTrue(prefer_initial(.4815949367451729, .4815949499607086))

    def test_backbone_training_no_longer_skips_lar(self):
        logits = torch.tensor([[0., 1.]])
        harness = SimpleNamespace(
            configs=SimpleNamespace(relook_backbone_only=True, relook_truth_set_weight=0.,
                                    n_lar=1, alpha_step=0.), alpha=.2,
            _shared_backbone_forward=lambda *a: (logits, None, None, None, None),
            loss_fn=lambda *a: torch.tensor(3.),
            _language_adversarial_loss=lambda *a: torch.tensor(2.), history={'loss': []})
        loss = ReLookKG.training_step(harness, dict(ent_rel=None, source_ids=None,
            source_mask=None, labels=torch.tensor([1]), lars=torch.tensor([[0]])), 0)['loss']
        self.assertAlmostEqual(loss.item(), 3.4, places=6)


if __name__ == '__main__':
    unittest.main()
