import unittest
from types import SimpleNamespace
import torch
from models.relook import UncertaintyGate, UtilityGate, JointEvidenceRouter, RelationRoleRouter
from models.relook_model import ReLookKG


class LockedPoliciesTest(unittest.TestCase):
    def test_fp16_extreme_confidence_is_finite(self):
        logits = torch.tensor([[1000, 0, -1000], [0, 0, 0]], dtype=torch.float16)
        for module in [UncertaintyGate, UtilityGate]:
            self.assertTrue(torch.isfinite(module.confidence_statistics(logits, 3)).all())
        query = torch.zeros(2, 8, dtype=torch.float16)
        delta = torch.full_like(query, 1000)
        self.assertTrue(torch.isfinite(RelationRoleRouter.query_statistics(logits, query, delta, 3)).all())
        self.assertTrue(torch.isfinite(JointEvidenceRouter._logit(torch.tensor([0.,1.], dtype=torch.float16))).all())

    def test_equal_quotas_and_repeatability(self):
        config = SimpleNamespace(selection_policy='random', selection_budget=.5, seed=41504, n_rel=5)
        harness = SimpleNamespace(configs=config)
        er = torch.tensor([[i, i%10] for i in range(12)])
        logits = torch.arange(120).reshape(12,10).float()
        probability = torch.linspace(0,1,12)
        eligible = torch.ones(12,dtype=torch.bool)
        eligible[-1] = False
        for policy in ['random','entropy','margin','learned']:
            config.selection_policy = policy
            a = ReLookKG._selection_mask(harness,probability,logits,er,eligible)
            b = ReLookKG._selection_mask(harness,probability,logits,er,eligible)
            self.assertEqual(int(a.sum()),6)
            self.assertTrue(torch.equal(a,b))
            self.assertFalse(a[-1])


if __name__ == '__main__':
    unittest.main()
