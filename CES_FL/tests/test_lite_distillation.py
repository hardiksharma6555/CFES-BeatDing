"""Independent numerical/gradient checks for proposed distillation primitives."""
import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ces_fl_lite_feasibility as lite


class DistillationTests(unittest.TestCase):
    def test_binary_kl_matches_two_class_reference_and_detaches_teacher(self):
        s = torch.tensor([-2., .3, 1.1], requires_grad=True)
        t = torch.tensor([-.5, 1., .2], requires_grad=True)
        cells = torch.tensor([0, 0, 1])
        value = lite.logit_distillation(s, t, cells, temperature=2.)
        teacher = F.softmax(torch.stack([torch.zeros_like(t), t.detach()], 1) / 2, 1)
        student = F.log_softmax(torch.stack([torch.zeros_like(s), s], 1) / 2, 1)
        per = F.kl_div(student, teacher, reduction='none').sum(1) * 4
        reference = .5 * (per[:2].mean() + per[2])
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)
        value.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertGreater(float(s.grad.abs().sum()), 0)

    def test_cell_balance_not_sample_weighted(self):
        loss = torch.tensor([1., 1., 1., 9.])
        cells = torch.tensor([0, 0, 0, 1])
        self.assertEqual(float(lite.balanced_mean(loss, cells)), 5.)

    def test_response_shift_invariant_and_teacher_detached(self):
        s = torch.tensor([.1, .2], requires_grad=True)
        t = torch.tensor([1., -.2], requires_grad=True)
        teacher_s = torch.tensor([-1., 2.], requires_grad=True)
        teacher_t = torch.tensor([.2, 2.3], requires_grad=True)
        groups = torch.tensor([0, 1])
        v = lite.response_distillation(s, t, teacher_s, teacher_t, groups)
        shifted = lite.response_distillation(s+5, t+5, teacher_s-3, teacher_t-3, groups)
        torch.testing.assert_close(v, shifted)
        v.backward()
        self.assertIsNone(teacher_s.grad); self.assertIsNone(teacher_t.grad)
        self.assertGreater(float(s.grad.abs().sum()+t.grad.abs().sum()), 0)

    def test_equal_logits_zero_kl_and_bad_inputs_rejected(self):
        x = torch.tensor([-3., .8])
        torch.testing.assert_close(lite.logit_distillation(x, x, torch.tensor([0, 1])), torch.tensor(0.))
        with self.assertRaises(ValueError): lite.logit_distillation(x, x, torch.tensor([0, 1]), 0)
        with self.assertRaises(ValueError): lite.balanced_mean(x[:0], torch.tensor([], dtype=torch.long))


if __name__ == '__main__': unittest.main()
