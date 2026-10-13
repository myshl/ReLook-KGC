import unittest
from types import SimpleNamespace

import torch

from models.relook_model import ReLookKG


class FilteredVisualUtilityTest(unittest.TestCase):
    def setUp(self):
        self.harness = SimpleNamespace(
            configs=SimpleNamespace(n_rel=1),
            train_tail_gt={(0, 0): [1, 2]},
            train_head_gt={},
        )
        self.ent_rel = torch.tensor([[0, 0]])
        self.label = torch.tensor([1])

    def filtered_rr(self, logits):
        return ReLookKG._training_filtered_reciprocal_rank(
            self.harness, logits, self.ent_rel, self.label
        )

    def test_alternative_true_answer_does_not_harm_rank(self):
        base = torch.tensor([[3.0, 4.0, 3.0]])
        visual = torch.tensor([[3.0, 4.0, 5.0]])

        raw_gain = (
            ReLookKG._raw_reciprocal_rank(visual, self.label)
            - ReLookKG._raw_reciprocal_rank(base, self.label)
        )
        filtered_gain = self.filtered_rr(visual) - self.filtered_rr(base)

        self.assertLess(raw_gain.item(), 0.0)
        self.assertEqual(filtered_gain.item(), 0.0)

    def test_filtered_rank_recovers_hidden_visual_gain(self):
        # Entity 2 is another true answer.  Raw rank sees no improvement because
        # it moves above the target, while filtered rank correctly sees the
        # remaining false candidate move below it.
        base = torch.tensor([[5.0, 4.0, 3.0]])
        visual = torch.tensor([[5.0, 5.5, 6.0]])

        raw_gain = (
            ReLookKG._raw_reciprocal_rank(visual, self.label)
            - ReLookKG._raw_reciprocal_rank(base, self.label)
        )
        filtered_gain = self.filtered_rr(visual) - self.filtered_rr(base)

        self.assertEqual(raw_gain.item(), 0.0)
        self.assertGreater(filtered_gain.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
