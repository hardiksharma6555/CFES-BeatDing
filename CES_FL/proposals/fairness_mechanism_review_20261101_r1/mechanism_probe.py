"""Synthetic mechanism probe, NOT a detector training or fairness benchmark.

Tests whether final-score alignment can miss demographic shifts in the joint
global/local contribution distribution. No data, checkpoints or downloads.
"""
import argparse
import json
from pathlib import Path

import torch


def mmd2(a, b, sigma=4.0):
    """Biased empirical Gaussian-kernel MMD²; diagonals included, differentiable."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Expected sample-by-feature matrices with matching features')
    if not len(a) or not len(b) or sigma <= 0:
        raise ValueError('Empty samples or invalid bandwidth')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite contributions')
    def kernel(x, y):
        return torch.exp(-((x[:, None] - y[None, :]) ** 2).sum(-1) / (2 * sigma ** 2))
    return kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean()


def conditional_joint_loss(evidence, labels, groups, sigma=4.0):
    """Equal class/group weights; reference is a group-balanced empirical mixture.

    Toy research implementation. Both classes require >=2 groups and >=2 samples
    per cell. No silent dropping; gradients include the reference distribution.
    Equivalent to the mean squared distance to the RKHS mean embedding.
    """
    if labels.shape != (len(evidence),) or groups.shape != labels.shape:
        raise ValueError('Misaligned labels/groups')
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Expected binary labels')
    class_terms = []
    for label in (0, 1):
        subset = evidence[labels == label]
        gs = groups[labels == label]
        cells = [subset[gs == g] for g in torch.unique(gs)]
        if len(cells) < 2 or any(len(cell) < 2 for cell in cells):
            raise ValueError('Insufficient class/group support')
        # 1/(2G²) sum_{g,h} ||mu_g - mu_h||² = 1/G sum_g ||mu_g - mean(mu)||².
        term = evidence.sum() * 0
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                term = term + mmd2(cells[i], cells[j], sigma)
        class_terms.append(term / len(cells) ** 2)
    return torch.stack(class_terms).mean()


def fixture(offset):
    # Every group has identical class-specific final scores by construction.
    scores = torch.tensor([-2., -1., -.4, .2, -.2, .4, 1., 2.], dtype=torch.float64)
    labels = torch.tensor([0] * 4 + [1] * 4)
    signed = torch.cat([offset.expand(8), -offset.expand(8)])
    z = scores.repeat(2)
    e = torch.stack([z / 2 + signed, z / 2 - signed], dim=1)
    return e, labels.repeat(2), torch.tensor([0] * 8 + [1] * 8)


def errors(e, y, g, rho):
    # Same perturbation and same threshold for every group; diagnostic, not
    # an assertion that compression/domain shift literally rescales branches.
    scores = (1 + rho) * e[:, 0] + (1 - rho) * e[:, 1]
    pred = scores >= 0
    fpr = [float(pred[(y == 0) & (g == k)].double().mean()) for k in (0, 1)]
    fnr = [float((~pred[(y == 1) & (g == k)]).double().mean()) for k in (0, 1)]
    return {'fpr': fpr, 'fnr': fnr, 'fpr_gap': max(fpr)-min(fpr), 'fnr_gap': max(fnr)-min(fnr)}


def probe():
    torch.set_num_threads(2)
    d = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    e, y, g = fixture(d)
    score_loss = conditional_joint_loss(e.sum(1, keepdim=True), y, g)
    joint_loss = conditional_joint_loss(e, y, g)
    score_grad = torch.autograd.grad(score_loss, d, retain_graph=True)[0]
    joint_grad = torch.autograd.grad(joint_loss, d)[0]
    assert abs(float(score_loss.detach())) < 1e-12 and abs(float(score_grad)) < 1e-12
    assert float(joint_loss.detach()) > .01 and abs(float(joint_grad)) > 1e-5
    initial = {'offset': float(d.detach()), 'score_mmd2': float(score_loss.detach()),
               'joint_mmd2': float(joint_loss.detach()), 'score_gradient': float(score_grad),
               'joint_gradient': float(joint_grad), 'nominal_errors': errors(e, y, g, 0),
               'perturbed_errors': errors(e, y, g, .1)}
    nominal_scores = e.sum(1).detach().clone()
    optimizer = torch.optim.Adam([d], lr=.1)
    for _ in range(200):
        e, y, g = fixture(d)
        loss = conditional_joint_loss(e, y, g)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    e, y, g = fixture(d)
    torch.testing.assert_close(e.sum(1), nominal_scores, rtol=0, atol=1e-12)
    assert abs(float(d.detach())) < .01
    assert errors(e, y, g, .1)['fpr_gap'] == 0
    assert errors(e, y, g, .1)['fnr_gap'] == 0
    # Equal marginal branch distributions do NOT imply equal joint evidence.
    a = torch.tensor([[-1., -1.], [1., 1.]], dtype=torch.float64)
    b = torch.tensor([[-1., 1.], [1., -1.]], dtype=torch.float64)
    marginal = [float(mmd2(a[:, k:k+1], b[:, k:k+1])) for k in (0, 1)]
    assert max(abs(v) for v in marginal) < 1e-12 and float(mmd2(a, b)) > 0
    # Numerical gradient verification on general samples, independently of fixture.
    gen = torch.Generator().manual_seed(17)
    random_e = torch.randn(8, 2, generator=gen, dtype=torch.float64, requires_grad=True)
    yy = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    gg = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    assert torch.autograd.gradcheck(lambda v: conditional_joint_loss(v, yy, gg), (random_e,))
    return {'status': 'PASS', 'data': 'constructed synthetic evidence, no faces or detector training',
            'initial': initial, 'after_toy_offset_optimization': {'offset': float(d.detach()),
                'joint_mmd2': float(conditional_joint_loss(e, y, g).detach()),
                'nominal_errors': errors(e, y, g, 0), 'perturbed_errors': errors(e, y, g, .1)},
            'marginal_vs_joint_example': {'marginal_mmd2': marginal, 'joint_mmd2': float(mmd2(a, b))},
            'gradcheck': True, 'real_checkpoint_cancellation_verified': False,
            'real_fairness_improvement_verified': False, 'novelty_established': False,
            'limitations': ['Toy nuisance offset is known and directly optimizable.',
                'No proof that real corruptions act as branch gain changes.',
                'Small empirical MMD does not certify threshold fairness.',
                'Evidence decomposition may change under retraining/reparameterization.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError('Preserve prior evidence: ' + str(path))
    result = probe()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
