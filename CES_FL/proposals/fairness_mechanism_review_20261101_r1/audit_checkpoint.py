"""Read-only evidence audit of a pinned TRAIN/validation checkpoint. Never trains.

Only aggregate results are exported. The shared threshold is taken unchanged
from the checkpoint. Gain perturbations are diagnostics, not image corruptions.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import ces_fl_upgrade as up
from mechanism_probe import conditional_joint_loss

VERSION = 'CESFL_EVIDENCE_AUDIT_v1'
UPGRADE_SHA = '4d8413cbfd8f22c60be8405ee7686dff7745aed134edddecf21e809cf3082ff6'


def summarize(rows, global_c, local_c, bias, threshold, c, original_logits=None):
    global_c, local_c = np.asarray(global_c, float), np.asarray(local_c, float)
    if global_c.shape != (len(rows),) or local_c.shape != global_c.shape:
        raise ValueError('Contribution/row mismatch')
    if not np.isfinite(global_c).all() or not np.isfinite(local_c).all():
        raise ValueError('Nonfinite contributions')
    if not math.isfinite(bias) or not math.isfinite(threshold):
        raise ValueError('Nonfinite bias/threshold')
    nominal = global_c + local_c + bias if original_logits is None else np.asarray(original_logits, float)
    if nominal.shape != global_c.shape or not np.isfinite(nominal).all():
        raise ValueError('Invalid nominal logits')
    if not np.allclose(nominal, global_c + local_c + bias, rtol=1e-5, atol=1e-5):
        raise ValueError('Nominal score decomposition differs')
    y = np.asarray([r['label'] for r in rows])
    group = np.asarray([r[c['axis']] for r in rows])
    known = sorted(set(group) - {'unknown', ''})
    eligible = [g for g in known if all(int(((group == g) & (y == k)).sum()) >= c['min_support'] for k in (0, 1))]
    mean_records = []
    for label in (0, 1):
        means = {g: (float(global_c[(group == g) & (y == label)].mean()),
                     float(local_c[(group == g) & (y == label)].mean())) for g in eligible}
        if not means:
            continue
        reference = np.asarray(list(means.values())).mean(0)
        for g, m in means.items():
            dg, dl = np.asarray(m) - reference
            mean_records.append({'class': label, 'group': g,
                'n': int(((group == g) & (y == label)).sum()),
                'global_mean': m[0], 'local_mean': m[1],
                'global_shift': float(dg), 'local_shift': float(dl),
                'total_shift': float(dg + dl),
                'cancellation_amount_logit_units': float(max(0, abs(dg) + abs(dl) - abs(dg + dl)))})
    kernel = {'status': 'INSUFFICIENT_GROUP_SUPPORT', 'eligible_groups': eligible}
    if len(eligible) >= 2:
        per_cell = min(c['kernel_samples_per_cell'], min(int(((group == g) & (y == k)).sum()) for g in eligible for k in (0, 1)))
        rng = np.random.default_rng(c['seed'])
        indices = np.concatenate([rng.choice(np.flatnonzero((group == g) & (y == k)), per_cell, replace=False)
                                  for k in (0, 1) for g in eligible])
        evidence = torch.tensor(np.stack([global_c[indices], local_c[indices]], 1), dtype=torch.float64)
        yy = torch.tensor(y[indices]); gg = torch.tensor([eligible.index(g) for g in group[indices]])
        kernel = {'status': 'COMPUTED', 'eligible_groups': eligible, 'samples_per_cell': per_cell,
            'estimator': 'biased empirical Gaussian MMD squared; equal class/group mixture; fixed bandwidth',
            'sigma_in_logit_units': c['kernel_sigma'],
            'score_mmd2': float(conditional_joint_loss(evidence.sum(1, keepdim=True), yy, gg, c['kernel_sigma'])),
            'joint_mmd2': float(conditional_joint_loss(evidence, yy, gg, c['kernel_sigma'])),
            'warning': 'Different kernels/dimensions: raw score/joint MMD magnitudes are not directly comparable or significance tests.'}
    stress = []
    for rho in c['gain_perturbations']:
        # Preserve exact emitted logits at rho=0: double-precision reconstruction
        # can otherwise flip a sample precisely on a float32 calibration boundary.
        z = nominal + rho * (global_c - local_c)
        report = up.base.metrics(rows, z, threshold, c['min_support'])
        stress.append({'rho': rho, 'threshold_logit': threshold,
                       'overall': report['overall'], 'axis': report['axes'][c['axis']]})
    return {'group_class_means': mean_records, 'kernel_diagnostic': kernel,
            'gain_stress': stress, 'groups_excluded_for_support': sorted(set(known)-set(eligible)),
            'unknown_group_rows': int(np.isin(group, ['unknown', '']).sum()),
            'cluster_count': len({r['cluster'] for r in rows}),
            'uncertainty': 'Descriptive only; cluster bootstrap and independent-domain validation remain required.'}


def validate_config(c):
    required = {'checkpoint','checkpoint_sha256','manifest','manifest_sha256','split',
                'axis','min_support','batch_size','workers','seed','kernel_sigma',
                'kernel_samples_per_cell','gain_perturbations'}
    if set(c) != required:
        raise ValueError('Unexpected/missing audit configuration keys')
    if c['split'] not in ('train', 'val') or c['axis'] not in up.base.AXES:
        raise ValueError('Only TRAIN/validation with a declared axis is allowed')
    for k in ('min_support','batch_size','kernel_samples_per_cell'):
        if type(c[k]) is not int or c[k] < 2:
            raise ValueError('Invalid ' + k)
    for k in ('workers','seed'):
        if type(c[k]) is not int or c[k] < 0:
            raise ValueError('Invalid ' + k)
    if not math.isfinite(c['kernel_sigma']) or c['kernel_sigma'] <= 0:
        raise ValueError('Invalid kernel bandwidth')
    gains = c['gain_perturbations']
    if not isinstance(gains, list) or not gains or 0 not in gains or any(not math.isfinite(v) or abs(v) >= 1 for v in gains):
        raise ValueError('Need nominal rho=0 and finite positive branch gains')


@torch.inference_mode()
def run(c, out, device):
    validate_config(c)
    # Check before model allocation and never overwrite an old audit.
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior output: ' + str(out))
    if device not in ('cpu','cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested device unavailable')
    if up.base.sha(up.__file__) != UPGRADE_SHA or up.base.sha(up.base.__file__) != up.BASE_SHA:
        raise ValueError('Model utility source changed')
    for key in ('checkpoint','manifest'):
        if up.base.sha(c[key]) != c[key + '_sha256']:
            raise ValueError('Changed input: ' + key)
    model, ck = up.load_model(c['checkpoint'], device)
    if ck['input_hashes'][c['split']] != c['manifest_sha256']:
        raise ValueError('Manifest is not the checkpoint original TRAIN/validation split')
    if ck['source_sha256'] != {'upgrade': UPGRADE_SHA, 'base': up.BASE_SHA}:
        raise ValueError('Checkpoint source provenance differs')
    rows = up.base.read_rows(c['manifest'])
    model.eval(); torch.set_num_threads(4)
    ds = up.base.PairImages(rows, size=ck['config']['size'], backbone=model.backbone)
    loader = DataLoader(ds, batch_size=c['batch_size'], shuffle=False,
                        num_workers=c['workers'], collate_fn=up.base.collate)
    gc, lc, logits = [], [], []
    max_decomposition_error = 0.
    for batch in loader:
        result = model(batch['x'].to(device))
        reconstructed = result['global_contribution'] + result['local_contribution'] + result['fusion_bias']
        torch.testing.assert_close(result['logit'], reconstructed, rtol=1e-5, atol=1e-5)
        max_decomposition_error = max(max_decomposition_error, float((result['logit']-reconstructed).abs().max()))
        gc.extend(result['global_contribution'].cpu().tolist())
        lc.extend(result['local_contribution'].cpu().tolist())
        logits.extend(result['logit'].cpu().tolist())
    threshold = ck['calibration']['threshold_logit']
    report = summarize(rows, gc, lc, float(model.fusion_bias), threshold, c, logits)
    report.update({'format': VERSION, 'split': c['split'], 'axis': c['axis'], 'config': c,
        'n': len(rows), 'device': device, 'checkpoint_epoch': ck['epoch'],
        'calibration': ck['calibration'], 'threshold_recalibrated': False,
        'source_sha256': {'audit': up.base.sha(__file__), 'probe': up.base.sha(Path(__file__).with_name('mechanism_probe.py')),
                          'upgrade': UPGRADE_SHA, 'base': up.BASE_SHA},
        'max_decomposition_error': max_decomposition_error,
        'synthetic_smoke': bool(ck['synthetic_smoke']),
        'training_started': False, 'test_read': False,
        'meaning': 'Mechanism audit only. Opposite contribution shifts are not proof of harmful bias or causal discrimination.'})
    nominal = up.base.metrics(rows, logits, threshold, c['min_support'])
    report['nominal_metrics'] = nominal
    if c['split'] == 'val':
        recorded = ck['validation']['overall']['auc']
        report['validation_auc_replay'] = {'recorded': recorded, 'replayed': nominal['overall']['auc'],
            'matches_at_1e_minus_5': abs(recorded - nominal['overall']['auc']) <= 1e-5}
    if up.base.sha(c['checkpoint']) != c['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during audit')
    out.mkdir(parents=True)
    up.base.write(out/'report.json', report)
    (out/'DONE').write_text('Read-only audit completed; no fairness improvement or novelty established.\n')
    print(json.dumps({'status': 'AUDIT_COMPLETE', 'n': len(rows), 'out': str(out), 'training_started': False}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    a = p.parse_args()
    run(json.loads(Path(a.config).read_text()), a.out, a.device)
