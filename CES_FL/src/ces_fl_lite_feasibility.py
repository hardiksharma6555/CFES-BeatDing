#!/usr/bin/env python3
"""Compact detector candidates and reproducible deployment-resource checks.

This is an architecture/serialization feasibility prototype, not a trained
detector. The profile command uses RANDOM weights and SYNTHETIC calibration.
Its exported files must not be used for detection. Host QNNPACK INT8 is not
Espressif's power-of-two quantization and is not an ESP32 deployment artifact.
No downloads, real data access, benchmark evaluation, or GPU jobs occur.

python ces_fl_lite_feasibility.py profile --out /tmp/cesfl_lite_profile
"""
import argparse
import copy
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
import torchvision


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel, stride=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride,
                                  kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


class MicroDetector(nn.Module):
    """Sequential depthwise-separable baseline; no claimed architectural novelty.

    Input: normalized RGB face crop [N,3,H,W]. Output: [N,1] fake logit.
    The face detector, cropper, camera buffers and image decoder are external.
    """
    def __init__(self):
        super().__init__()
        blocks = [ConvBNReLU(3, 16, 3, 2)]
        cin = 16
        for cout, stride in [(24, 1), (32, 2), (48, 1), (64, 2), (96, 1),
                             (128, 2), (160, 1), (192, 2), (192, 1)]:
            blocks.extend([ConvBNReLU(cin, cin, 3, stride, groups=cin),
                           ConvBNReLU(cin, cout, 1)])
            cin = cout
        self.quant = QuantStub()
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cin, 1)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.features(self.quant(x))
        x = self.pool(x).flatten(1)
        return self.dequant(self.classifier(x))

    def fuse_model(self):
        for layer in self.features:
            torch.ao.quantization.fuse_modules(layer, ['0', '1', '2'], inplace=True)


CANDIDATES = {
    'mobile_v2_relu_100': {'size': 224, 'width_mult': 1.0},
    'mobile_v2_relu_050': {'size': 160, 'width_mult': .5},
    'micro_ds_096': {'size': 96, 'width_mult': None},
}


def make_model(name):
    c = CANDIDATES[name]
    if c['width_mult'] is None:
        return MicroDetector()
    from torchvision.models.quantization import mobilenet_v2
    # The quantizable torchvision variant substitutes ReLU for ReLU6.
    model = mobilenet_v2(weights=None, quantize=False, width_mult=c['width_mult'])
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def balanced_mean(loss, cells):
    """Equal weighting of present class-by-group cells; eligibility is external."""
    if loss.ndim != 1 or cells.shape != loss.shape or not loss.numel():
        raise ValueError('Require nonempty aligned scalar losses and cell codes')
    return torch.stack([loss[cells == g].mean() for g in torch.unique(cells)]).mean()


def logit_distillation(student, teacher, cells, temperature=2.):
    """Group-balanced binary KL(teacher || student), with detached teacher.

    A tested loss primitive for a PROPOSED training study; not an accuracy result.
    Ground-truth BCE and direct fairness objectives must remain in that study.
    """
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('temperature must be positive and finite')
    if student.shape != teacher.shape or student.ndim != 1:
        raise ValueError('Logits must be aligned one-dimensional tensors')
    s, t = student.float() / temperature, teacher.detach().float() / temperature
    q = torch.sigmoid(t)
    per = q * (F.logsigmoid(t) - F.logsigmoid(s))
    per += (1 - q) * (F.logsigmoid(-t) - F.logsigmoid(-s))
    return temperature ** 2 * balanced_mean(per, cells)


def response_distillation(student_sham, student_treated, teacher_sham,
                          teacher_treated, groups):
    """Match asinh(logit difference) on the SAME registered procedural pairs.

    One operator/strength stratum per call. Cannot establish causal validity.
    """
    tensors = [student_sham, student_treated, teacher_sham, teacher_treated]
    if any(t.ndim != 1 or t.shape != tensors[0].shape for t in tensors):
        raise ValueError('Responses require aligned one-dimensional tensors')
    sr = torch.asinh(student_treated.float() - student_sham.float())
    tr = torch.asinh(teacher_treated.detach().float() - teacher_sham.detach().float())
    return balanced_mean(F.smooth_l1_loss(sr, tr, reduction='none'), groups)


@torch.inference_mode()
def resource_counts(model, x):
    """Count convolution/linear MACs and largest single leaf output.

    Largest output is a tensor-size statistic, NOT peak working RAM: live
    tensors, kernel scratch, allocator, inputs, weights and firmware add memory.
    """
    macs, largest, handles = [0], [0], []

    def hook(m, args, output):
        if torch.is_tensor(output):
            largest[0] = max(largest[0], output.numel())
            if isinstance(m, nn.Conv2d):
                macs[0] += output.numel() * (m.in_channels // m.groups) * np.prod(m.kernel_size)
            elif isinstance(m, nn.Linear):
                macs[0] += output.numel() * m.in_features

    for m in model.modules():
        if not list(m.children()):
            handles.append(m.register_forward_hook(hook))
    try:
        model(x)
    finally:
        for h in handles: h.remove()
    return {'conv_linear_macs': int(macs[0]),
            'largest_leaf_output_elements': largest[0],
            'largest_leaf_output_fp32_bytes': 4 * largest[0],
            'largest_leaf_output_hypothetical_int8_bytes': largest[0],
            'peak_runtime_ram_measured': False}


@torch.inference_mode()
def time_model(model, x, repeats=30):
    for _ in range(5): model(x)
    ms = []
    for _ in range(repeats):
        begin = time.perf_counter()
        model(x)
        ms.append(1000 * (time.perf_counter() - begin))
    return {'p50_ms': float(np.median(ms)), 'p95_ms': float(np.quantile(ms, .95)),
            'repeats': repeats, 'includes_face_detection_decode_crop': False,
            'device': 'host_cpu_NOT_ESP32'}


def export_script(model, path, sample):
    scripted = torch.jit.freeze(torch.jit.script(model.eval()))
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path)).eval()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(sample), model(sample), rtol=1e-5, atol=1e-6)
    return loaded


def audit_observers(model):
    """Permit only torchvision's unused skip observer on non-residual blocks."""
    unused = []
    for name, ob in model.named_modules():
        if not hasattr(ob, 'min_val') or not hasattr(ob, 'max_val'):
            continue
        observed = ob.min_val.numel() > 0 and torch.isfinite(ob.min_val).all()
        if observed:
            continue
        suffix = '.skip_add.activation_post_process'
        if not name.endswith(suffix):
            raise RuntimeError('Uncalibrated live observer: ' + name)
        parent = model.get_submodule(name[:-len(suffix)])
        if not hasattr(parent, 'use_res_connect') or parent.use_res_connect:
            raise RuntimeError('Uncalibrated live residual observer: ' + name)
        unused.append(name)
    return unused


def profile(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory; prior results are preserved')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    if 'qnnpack' not in torch.backends.quantized.supported_engines:
        raise RuntimeError('This reproducible host check requires QNNPACK')
    torch.backends.quantized.engine = 'qnnpack'
    report = {'status': 'SOFTWARE_RESOURCE_CHECK_ONLY', 'trained': False,
              'calibration': 'SYNTHETIC_RANDOM_NOT_PRODUCTION',
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'platform': platform.platform(), 'threads': 1, 'quant_backend': 'qnnpack',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'decimal_mb_divisor': 1000000,
              'esp32_artifact_exported': False, 'esp32_execution_verified': False,
              'accuracy_or_fairness_evaluated': False, 'models': {}}
    for name, c in CANDIDATES.items():
        torch.manual_seed(17)
        model = make_model(name)
        size = c['size']
        # Exercise gradients with synthetic labels; this is not task training.
        xb = 2 * torch.rand(4, 3, size, size) - 1
        yb = torch.tensor([0., 1., 0., 1.])
        loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
        model.zero_grad(set_to_none=True)
        model.eval()
        x = xb[:1]
        count = sum(p.numel() for p in model.parameters())
        row = {'input_shape': list(x.shape), 'parameters': count,
               'raw_fp32_parameter_bytes': count * 4,
               'raw_int8_parameter_bytes_estimate': count,
               'all_trainable_parameters_have_finite_gradients': True,
               **resource_counts(model, x)}
        fp = out / (name + '_RANDOM_FP32.pt')
        scripted_fp = export_script(model, fp, x)
        row['fp32_file_bytes'] = fp.stat().st_size
        row['fp32_host_timing'] = time_model(scripted_fp, x)
        q = copy.deepcopy(model).eval()
        q.fuse_model()
        with torch.inference_mode():
            torch.testing.assert_close(q(x), model(x), atol=1e-5, rtol=1e-4)
        q.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        torch.ao.quantization.prepare(q, inplace=True)
        with torch.inference_mode():
            for _ in range(8): q(2 * torch.rand(1, 3, size, size) - 1)
        row['unused_nonresidual_skip_observers'] = audit_observers(q)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            torch.ao.quantization.convert(q, inplace=True)
        row['conversion_warnings'] = sorted({str(w.message) for w in caught})
        if not any(isinstance(m, torch.ao.nn.quantized.Conv2d) for m in q.modules()):
            raise AssertionError('No genuinely quantized convolution was produced')
        with torch.inference_mode():
            pred = q(x)
            assert pred.shape == (1, 1) and torch.isfinite(pred).all()
        qp = out / (name + '_RANDOM_QNNPACK_INT8.pt')
        scripted_q = export_script(q, qp, x)
        row['int8_file_bytes'] = qp.stat().st_size
        row['int8_host_timing'] = time_model(scripted_q, x)
        row['saved_reloaded_fp32_and_int8_outputs_match'] = True
        row['quantized_conv_modules'] = sum(isinstance(m, torch.ao.nn.quantized.Conv2d) for m in q.modules())
        report['models'][name] = row
        (out / 'RESOURCE_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'model': name, 'params': count,
                          'fp32_MB': row['fp32_file_bytes'] / 1e6,
                          'int8_MB': row['int8_file_bytes'] / 1e6,
                          'MACs_M': row['conv_linear_macs'] / 1e6}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('profile'); p.add_argument('--out', required=True)
    args = parser.parse_args()
    profile(args.out)
