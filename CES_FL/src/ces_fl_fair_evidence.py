#!/usr/bin/env python3
"""Standalone CES-FL fair-evidence experiment, v1, 1 November 2026.

Requires Python >=3.10, torch, numpy, Pillow, scikit-learn (torchvision only for
ResNet). No project imports, network access, downloads or automatic training.

Implements original-image clean/JPEG/resize conditions, exact gated global/local
contributions, class-conditioned joint MMD, worst-condition soft FPR/FNR gaps,
matched BCE/score/marginal/joint/full ablations, validation selection/calibration,
checkpoint evaluation, contribution export, cancellation audit and smoke tests.
This is an experimental implementation, not proven novelty or improved fairness.
Generated demographic edits, forensic sensitivity and map consistency are absent.

Commands: init-config, preflight, dry-run, smoke, train, evaluate, export, audit.
Run --help or a command --help. Use fresh output directories. Training uses only
TRAIN/validation; evaluate is a separate explicit command. Selection and all
hyperparameters must be fixed before test. Gates are learned weights, not
reliability estimates; maps are signed local contributions, not forgery masks.

Embedded architecture/utilities come from CES-FL v0.2.1 (SHA256
0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de) and the
v0.3 partial-encoder class (4d8413cbfd8f22c60be8405ee7686dff7745aed134edddecf21e809cf3082ff6).
Xception classes originate from the vendored Ding utils/xception.py implementation;
its attribution is retained here. Checkpoint data remains external. New code below
is self-contained; copies do not import or modify historical experiment files.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

VERSION = 'CESFL_FAIR_EVIDENCE_v1'
PRETRAIN_FORMAT = 'CESFL_DIAGRAM_v0.2.1'
DEFAULT_XCEPTION = None
AXES = ('gender','race','age_bin','skin_bin','gender_race','ids8')
CONDITIONS = ('clean','jpeg','resize')

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(2048, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.last_linear = nn.Linear(512, self.num_classes)
        # else:
        #     self.last_linear = nn.Linear(2048, self.num_classes)

        # dag
        self.last_linear = nn.Linear(512, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )




        # if dropout:
        #     self.last_linear = nn.Sequential(
        #         nn.Dropout(p=dropout),
        #         nn.Linear(2048, self.num_classes)
        #     )

    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            # print(x.shape)
            x = self.block12(x)
            # print(x.shape, 12)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)

        return x

    def classifier(self, features):
        # x = self.relu(features) #([128, 512, 7, 7])
        x = F.adaptive_avg_pool2d(features, (1, 1)) #([128, 512, 1, 1])
        x = x.view(x.size(0), -1) #([128, 512])
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


class DiagramCES(nn.Module):
    def __init__(self,backbone='xception',weights=DEFAULT_XCEPTION,hidden=128,gate_hidden=64,pretrained=True,encoder_policy='frozen',bn_policy='frozen',head_seed=None):
        super().__init__();self.backbone=backbone;self.hidden=hidden;self.gate_hidden=gate_hidden
        if backbone=='xception':
            self.encoder=Xception({'mode':'original','num_classes':2,'inc':3,'dropout':False})
            if pretrained:
                if not Path(weights).is_file():raise FileNotFoundError(weights)
                sd=torch.load(weights,map_location='cpu',weights_only=True)
                sd={k:(v.unsqueeze(-1).unsqueeze(-1) if 'pointwise' in k and v.ndim==2 else v) for k,v in sd.items() if 'fc' not in k}
                result=self.encoder.load_state_dict(sd,strict=False)
                if result.unexpected_keys:raise ValueError(result.unexpected_keys)
                if any(not k.startswith(('adjust_channel.','last_linear.')) for k in result.missing_keys):raise ValueError(result.missing_keys)
            # Use pretrained2048-channel features, removing the unused RANDOM Ding512 adaptation.
            # This explicitly differs from v0.1; it avoids freezing random untrained adaptation weights.
            self.encoder.adjust_channel=nn.Identity();self.encoder.last_linear=nn.Identity();channels=2048
        elif backbone=='resnet50':
            from torchvision.models import resnet50
            m=resnet50(weights=None)
            if pretrained:
                if not weights or not Path(weights).is_file():raise FileNotFoundError('Supply a local ResNet50 ImageNet checkpoint via --weights')
                m.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
            self.encoder=nn.Sequential(*list(m.children())[:-2]);channels=2048
        else:raise ValueError('Supported encoders: xception,resnet50')
        self.encoder.requires_grad_(False)
        self.global_head=nn.Sequential(nn.Linear(channels,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.patch_head=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.ReLU(),nn.Conv2d(hidden,1,1))
        self.image_gate=nn.Sequential(nn.Linear(channels,gate_hidden),nn.ReLU(),nn.Linear(gate_hidden,1),nn.Sigmoid())
        self.patch_gate=nn.Sequential(nn.Conv2d(channels,gate_hidden,1),nn.ReLU(),nn.Conv2d(gate_hidden,1,1),nn.Sigmoid())
        self.fusion_raw=nn.Parameter(torch.full((2,),math.log(math.expm1(1.))))
        self.fusion_bias=nn.Parameter(torch.zeros(()))
        if head_seed is not None:self.reset_prediction_modules(head_seed)
        self.set_encoder_policy(encoder_policy,bn_policy)

    def reset_prediction_modules(self,seed):
        # A/B/C reset only prediction modules; preserve all encoder parameters/buffers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for name in ('global_head','patch_head','image_gate','patch_gate'):
                for module in getattr(self,name).modules():
                    if isinstance(module,(nn.Linear,nn.Conv2d)):module.reset_parameters()
            with torch.no_grad():
                self.fusion_raw.fill_(math.log(math.expm1(1.)));self.fusion_bias.zero_()

    def set_encoder_policy(self,encoder_policy,bn_policy):
        if encoder_policy not in ('frozen','trainable') or bn_policy not in ('frozen','update'):
            raise ValueError('Unknown encoder/BatchNorm policy')
        if encoder_policy=='frozen' and bn_policy!='frozen':raise ValueError('Frozen encoder requires frozen BN')
        self.encoder_policy,self.bn_policy=encoder_policy,bn_policy
        self.encoder.requires_grad_(encoder_policy=='trainable')
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if self.encoder_policy=='frozen':self.encoder.eval()
        elif self.bn_policy=='frozen':
            for module in self.encoder.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def forward(self,x):
        with (torch.no_grad() if self.encoder_policy=='frozen' else nullcontext()):
            spatial=self.encoder.features(x) if self.backbone=='xception' else self.encoder(x)
        global_features=spatial.mean((2,3))
        zg=self.global_head(global_features).squeeze(1);zp=self.patch_head(spatial).squeeze(1)
        rg=self.image_gate(global_features).squeeze(1);rp=self.patch_gate(spatial).squeeze(1)
        weights=F.softplus(self.fusion_raw)
        global_contribution=weights[0]*rg*zg;local_map=weights[1]*rp*zp
        local_contribution=local_map.mean((1,2));z=global_contribution+local_contribution+self.fusion_bias
        return {'logit':z,'global_score':zg,'patch_score':zp,'image_gate':rg,'patch_gate':rp,
            'local_map':local_map,'global_contribution':global_contribution,'local_contribution':local_contribution,
            'fusion_bias':self.fusion_bias,'fusion_weights':weights}

    def counts(self):return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}


def read_rows(path):
    path=Path(path)
    with path.open(newline='') as f:
        r=csv.DictReader(f)
        if not {'image','label','identity_ids','source_ids'}.issubset(r.fieldnames or []):raise ValueError('Missing manifest image/label/identity_ids/source_ids')
        rows=list(r)
    images=set()
    for row in rows:
        p=Path(row['image']);p=(path.parent/p).resolve() if not p.is_absolute() else p.resolve();row['image']=str(p)
        if row['label'] not in ('0','1') or not p.is_file():raise ValueError(f'Invalid label/file: {row}')
        row['label']=int(row['label'])
        if str(p) in images:raise ValueError('Duplicate original image')
        images.add(str(p))
        if not row['identity_ids'].strip() or not row['source_ids'].strip():raise ValueError('Missing contributor provenance')
        for axis in AXES:row[axis]=row.get(axis) or 'unknown'
        row['cluster']=row.get('cluster') or row['identity_ids'].split(';')[0];row['family']=row.get('family') or 'unknown'
    if {r['label'] for r in rows}!={0,1}:raise ValueError('Both classes required')
    return rows


def check_splits(train,val,declared_file=None):
    declared=json.loads(Path(declared_file).read_text())['declared_overlap_values'] if declared_file else {}
    if {r['image'] for r in train}&{r['image'] for r in val}:raise ValueError('Image-path leakage')
    observed={}
    for field in ('identity_ids','source_ids'):
        a={v for r in train for v in r[field].split(';')};b={v for r in val for v in r[field].split(';')}
        overlap=sorted(a&b);key=field+':train:val'
        if overlap!=declared.get(key,[]):raise ValueError(f'Unrecorded {field} overlap or changed benchmark inventory')
        observed[field]=len(overlap)
    return {'overlap_counts':observed,'exact_declared_benchmark_exception':bool(declared),'person_ground_truth_verified':False}


def calibrate(real,target=.1):
    z=np.asarray(real,dtype=np.float64)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all() or not 0<=target<1:raise ValueError('Invalid calibration')
    f=Fraction(str(target))*len(z);budget=f.numerator//f.denominator
    threshold=float(np.nextafter(np.sort(z)[::-1][budget],np.inf))
    if not math.isfinite(threshold):raise ValueError('Cannot represent finite threshold')
    fp=int((z>=threshold).sum())
    return {'threshold_logit':threshold,'rule':'validation_real_empirical_budget_v1','requested_fpr':target,'n_validation_real':len(z),
        'allowed_fp':budget,'achieved_fp':fp,'achieved_fpr':fp/len(z),'test_fpr_guaranteed':False}


def metrics(rows,logits,threshold,min_support=20):
    from sklearn.metrics import roc_auc_score
    z=np.asarray(logits);y=np.asarray([r['label'] for r in rows]);pred=z>=threshold
    if len(z)!=len(rows) or not np.isfinite(z).all():raise ValueError('Invalid logits')
    def one(mask):
        yy,zz,pp=y[mask],z[mask],pred[mask];real,fake=yy==0,yy==1
        return {'n':int(len(yy)),'n_real':int(real.sum()),'n_fake':int(fake.sum()),
            'auc':float(roc_auc_score(yy,zz)) if real.any() and fake.any() else None,
            'fpr':float(pp[real].mean()) if real.any() else None,'fnr':float((~pp[fake]).mean()) if fake.any() else None,
            'positive_rate':float(pp.mean()) if len(pp) else None}
    overall=one(np.ones(len(y),bool));axes={}
    for axis in AXES:
        labels=np.asarray([r.get(axis,'unknown') for r in rows]);groups={str(g):one(labels==g) for g in np.unique(labels)}
        good={g:r for g,r in groups.items() if g!='unknown' and min(r['n_real'],r['n_fake'])>=min_support}
        axes[axis]={'groups':groups,'eligible_groups':list(good),'min_support_per_class':min_support,
            'worst_group_auc':min(r['auc'] for r in good.values()) if good else None}
        for key in ('auc','fpr','fnr'):
            values=[r[key] for r in good.values()];axes[axis][key+'_gap']=max(values)-min(values) if len(values)>1 else None
        axes[axis]['ding_style_fractions']={'F_FPR':sum(abs(r['fpr']-overall['fpr']) for r in good.values()),
            'F_DP':max(r['positive_rate'] for r in good.values())-min(r['positive_rate'] for r in good.values()),
            'es_AUC':overall['auc']/(1+sum(abs(r['auc']-overall['auc']) for r in good.values()))} if len(good)>1 else None
    families=np.asarray([r.get('family','unknown') for r in rows]);by_family={}
    for family in np.unique(families):
        mask=families==family if family=='real' else (y==0)|((y==1)&(families==family))
        by_family[str(family)]=one(mask)
    return {'units':'fractions','threshold_logit':float(threshold),'overall':overall,'axes':axes,'by_family':by_family}


def build_optimizer(model,c):
    encoder=[p for p in model.encoder.parameters() if p.requires_grad]
    other=[p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    groups=[]
    if encoder:groups.append({'params':encoder,'lr':c.get('encoder_lr',c['lr']),'name':'encoder'})
    if other:groups.append({'params':other,'lr':c['lr'],'name':'heads_gates_fusion'})
    if not groups:raise ValueError('No trainable parameters')
    if not all(math.isfinite(g['lr']) and g['lr']>0 for g in groups):raise ValueError('Invalid learning rate')
    return torch.optim.AdamW(groups,lr=c['lr'],weight_decay=c['weight_decay'])


def import_encoder(model,c):
    path=c['encoder_checkpoint'];expected=c.get('encoder_checkpoint_sha256')
    if not expected or sha(path)!=expected:raise ValueError('Encoder checkpoint SHA256 missing/mismatched')
    ck=torch.load(path,map_location='cpu',weights_only=True);pc=ck['config']
    if ck.get('format')!=PRETRAIN_FORMAT or pc.get('stage')!='pretrain' or pc.get('encoder_policy')!='trainable' or pc['mode']!='bce':
        raise ValueError('Need a verified original-TRAIN BCE encoder-pretraining checkpoint')
    for key in ('backbone','size'):
        if pc[key]!=c[key]:raise ValueError('Encoder/preprocessing mismatch: '+key)
    if ck['training_inputs']!={'train_sha256':sha(c['train']),'val_sha256':sha(c['val'])}:
        raise ValueError('Pretraining/downstream split hashes differ')
    model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in ck['model'].items() if k.startswith('encoder.')},strict=True)
    return {'path':str(path),'sha256':expected,'epoch':ck['epoch'],'imported':'encoder only; fresh identical downstream heads/gates/fusion'}


def diverging(a,scale):
    t=np.clip(np.asarray(a,dtype=float)/max(scale,1e-12),-1,1);out=np.ones((*t.shape,3),float)
    out[...,0]=1-np.maximum(-t,0);out[...,1]=1-abs(t);out[...,2]=1-np.maximum(t,0)
    return (out*255).astype(np.uint8)


class UpgradeCES(DiagramCES):
    def __init__(self, backbone='xception', weights=None, hidden=128, gate_hidden=64,
                 pretrained=True, encoder_policy='partial', head_seed=17):
        super().__init__(backbone, weights, hidden, gate_hidden, pretrained,
                         encoder_policy='frozen', bn_policy='frozen', head_seed=head_seed)
        self.configure(encoder_policy)

    def configure(self, policy):
        if policy not in ('frozen', 'partial'):
            raise ValueError('Upgrade supports frozen or partial encoder only')
        self.encoder.requires_grad_(False)
        # Xception exit flow; ResNet layer4 (index 7 in the feature Sequential).
        self.selected_names = (('block12', 'conv3', 'bn3', 'conv4', 'bn4')
                               if self.backbone == 'xception' else ('7',))
        if policy == 'partial':
            for name in self.selected_names:
                self.encoder.get_submodule(name).requires_grad_(True)
        self.encoder_policy = policy
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Freeze BN running buffers throughout; affine parameters in the selected
        # stage may train. Early encoder modules remain in evaluation mode.
        self.encoder.eval()
        if getattr(self, 'encoder_policy', 'frozen') == 'partial':
            for name in self.selected_names:
                self.encoder.get_submodule(name).train(mode)
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


# New fair-evidence training implementation follows.

def fresh_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def default_config(arm='full'):
    root = '/home/sachin.chaudhary/ces_fl_pipeline'
    c = dict(seed=17, head_seed=17, backbone='xception', hidden=128, gate_hidden=64,
        size=224, epochs=10, workers=4, samples_per_cell=2, expected_groups=8,
        steps_per_epoch=0, forward_batch_size=16, eval_batch_size=16,
        encoder_policy='partial', axis='gender_race', min_train_cell=20, min_support=20,
        lr=.001, encoder_lr=.00001, weight_decay=.0001, jpeg_quality=85, resize_scale=.75,
        alignment='joint', lambda_align=.1, lambda_condition=.1, kernel_sigma=4.,
        train_threshold=0., soft_temperature=1., target_fpr=.1,
        selection='fairness_with_auc_floor', auc_floor=.95, worst_group_auc_floor=.85,
        train=root+'/manifests/ding_ffpp/train.csv', val=root+'/manifests/ding_ffpp/val.csv',
        protocol_overlap_file=root+'/manifests/ding_ffpp/protocol_source_overlap.json',
        encoder_checkpoint=root+'/audits/encoder_controls_20261030T202634/full_experiment_runs/P_pretrain/best.pt',
        encoder_checkpoint_sha256='484f09d6f826aa4baf1e4d6413813aa8608e320349082be377617c9f943e9dc8',
        input_hashes={'train':'6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657',
            'val':'52f4844cffe5a03adb08cecde8829244bb50e05eeead36f6d659599f1c24f5b1',
            'protocol_overlap_file':'793db56c30c85dd654c6d68745afc926f4b1f8dc8bec749db01b07b3aa306c97',
            'encoder_checkpoint':'484f09d6f826aa4baf1e4d6413813aa8608e320349082be377617c9f943e9dc8'},
        arm=arm, purpose='Exploratory matched fairness experiment; no novelty/performance claim')
    modes = {'bce':('none',0.,0.), 'score':('score',.1,0.),
             'marginal':('marginal',.1,0.), 'joint':('joint',.1,0.), 'full':('joint',.1,.1)}
    if arm not in modes:
        raise ValueError('Unknown arm')
    c['alignment'], c['lambda_align'], c['lambda_condition'] = modes[arm]
    return c


def validate_config(c):
    if set(c) != set(default_config()):
        raise ValueError(f'Config keys differ: missing={set(default_config())-set(c)}, extra={set(c)-set(default_config())}')
    for key in ('hidden','gate_hidden','epochs','samples_per_cell','expected_groups',
                'forward_batch_size','eval_batch_size','min_train_cell','min_support'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Invalid '+key)
    if c['samples_per_cell'] < 2 or c['expected_groups'] < 2:
        raise ValueError('At least two samples per class/group and two groups required')
    for key in ('seed','head_seed','workers','steps_per_epoch'):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError('Invalid '+key)
    if type(c['size']) is not int or c['size'] < 75:
        raise ValueError('Image size must be >=75')
    if c['backbone'] not in ('xception','resnet50') or c['encoder_policy'] not in ('frozen','partial'):
        raise ValueError('Unsupported backbone/policy')
    if c['axis'] not in AXES or c['alignment'] not in ('none','score','marginal','joint'):
        raise ValueError('Unsupported axis/alignment')
    if c['selection'] not in ('validation_auc_only','fairness_with_auc_floor'):
        raise ValueError('Invalid selection')
    for key in ('lr','encoder_lr','kernel_sigma','soft_temperature'):
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid '+key)
    for key in ('weight_decay','lambda_align','lambda_condition'):
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError('Invalid '+key)
    for key in ('target_fpr','auc_floor','worst_group_auc_floor'):
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or not 0 <= c[key] < 1:
            raise ValueError('Invalid '+key)
    if not math.isfinite(c['train_threshold']):
        raise ValueError('Invalid training threshold')
    if type(c['jpeg_quality']) is not int or not 60 <= c['jpeg_quality'] <= 100:
        raise ValueError('JPEG outside screened mild range [60,100]')
    if not math.isfinite(c['resize_scale']) or not .5 <= c['resize_scale'] < 1:
        raise ValueError('Resize scale outside screened range [.5,1)')
    if (c['alignment']=='none') != (c['lambda_align']==0):
        raise ValueError('Alignment none requires zero lambda, active alignment requires positive lambda')
    required = {'train','val','encoder_checkpoint'}
    if c['protocol_overlap_file']:
        required.add('protocol_overlap_file')
    if set(c['input_hashes']) != required:
        raise ValueError('Exact hashes required for every consumed external input')
    for key in required:
        digest = c['input_hashes'][key]
        if not isinstance(digest,str) or len(digest)!=64 or any(ch not in '0123456789abcdef' for ch in digest):
            raise ValueError('Invalid input SHA256: '+key)
    if c['encoder_checkpoint_sha256'] != c['input_hashes']['encoder_checkpoint']:
        raise ValueError('Encoder hash disagreement')


def cell_inventory(rows, c):
    groups = sorted({r[c['axis']] for r in rows})
    if any(g.strip().lower() in ('','unknown','nan','none') for g in groups):
        raise ValueError('Unknown primary demographic groups: do not silently relabel/drop rows')
    if len(groups) != c['expected_groups']:
        raise ValueError(f'Expected {c["expected_groups"]} groups, found {groups}')
    lookup = {g:i for i,g in enumerate(groups)}
    cells = defaultdict(list)
    for i,r in enumerate(rows):
        cells[(r['label'],lookup[r[c['axis']]])].append(i)
    for y in (0,1):
        for g in range(len(groups)):
            if len(cells[(y,g)]) < max(c['samples_per_cell'],c['min_train_cell']):
                raise ValueError(f'Insufficient TRAIN support for class={y}, group={groups[g]}')
    return groups, dict(cells)


def preflight(c):
    validate_config(c)
    for key,digest in c['input_hashes'].items():
        if sha(c[key]) != digest:
            raise ValueError('Changed input: '+key)
    rows, val = read_rows(c['train']), read_rows(c['val'])
    splits = check_splits(rows,val,c['protocol_overlap_file'])
    groups,cells = cell_inventory(rows,c)
    if c['selection']=='fairness_with_auc_floor':
        counts = Counter((r['label'],r[c['axis']]) for r in val)
        for y in (0,1):
            for g in groups:
                if counts[(y,g)] < c['min_support']:
                    raise ValueError('Validation support too small for fixed primary groups: '+str((y,g)))
    ck = torch.load(c['encoder_checkpoint'],map_location='cpu',weights_only=True)
    pc = ck['config']
    if ck.get('format')!=PRETRAIN_FORMAT or pc.get('stage')!='pretrain' or pc.get('encoder_policy')!='trainable' or pc.get('mode')!='bce':
        raise ValueError('Need original-TRAIN BCE forensic pretraining checkpoint')
    if any(pc[k]!=c[k] for k in ('backbone','size')):
        raise ValueError('Pretraining backbone/preprocessing mismatch')
    if ck['training_inputs']!={'train_sha256':c['input_hashes']['train'],'val_sha256':c['input_hashes']['val']}:
        raise ValueError('Pretraining and downstream split hashes differ')
    batch_size = len(cells)*c['samples_per_cell']
    summary = {'status':'PASS','train_rows':len(rows),'validation_rows':len(val),'groups':groups,
        'cell_counts':{f'{y}:{groups[g]}':len(ix) for (y,g),ix in sorted(cells.items())},
        'originals_per_step':batch_size,'image_forwards_per_step':batch_size*len(CONDITIONS),
        'steps_per_epoch':c['steps_per_epoch'] or math.ceil(len(rows)/batch_size),
        'splits':splits,'conditions':list(CONDITIONS),'training_started':False,
        'condition_validity':'Mild fixed engineering settings; no independent forensic-preservation certification'}
    return rows,val,groups,cells,summary


class BalancedCells(Sampler):
    """Every step has k distinct rows per class/group; repeats across steps logged."""
    def __init__(self,cells,k,steps,seed,epoch):
        self.cells,self.k,self.steps,self.seed,self.epoch = cells,k,steps,seed,epoch
        if k<2 or steps<1 or any(len(ix)<k for ix in cells.values()):
            raise ValueError('Invalid balanced sampling')
    def __len__(self):
        return self.steps
    def __iter__(self):
        rng = random.Random(f'{self.seed}:cells:{self.epoch}')
        for _ in range(self.steps):
            chosen = []
            for key in sorted(self.cells):
                chosen.extend(rng.sample(self.cells[key],self.k))
            rng.shuffle(chosen)
            yield chosen


def image_conditions(im,c):
    clean = im.convert('RGB').resize((c['size'],c['size']),Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    clean.save(buffer,format='JPEG',quality=c['jpeg_quality'],subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        jpeg = decoded.convert('RGB').copy()
    small = max(1,round(c['size']*c['resize_scale']))
    resized = clean.resize((small,small),Image.Resampling.BILINEAR).resize(clean.size,Image.Resampling.BILINEAR)
    return [clean,jpeg,resized]


def tensor_image(im,backbone):
    a = np.asarray(im,dtype=np.float32).copy()/255.
    x = torch.from_numpy(a).permute(2,0,1)
    if backbone=='xception':
        return (x-.5)/.5
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class ConditionImages(Dataset):
    def __init__(self,rows,c,groups=None):
        self.rows,self.c = rows,c
        self.lookup = {g:i for i,g in enumerate(groups or [])}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        r = self.rows[i]
        with Image.open(r['image']) as im:
            images = image_conditions(im,self.c)
        return {'images':torch.stack([tensor_image(im,self.c['backbone']) for im in images]),
                'y':torch.tensor(r['label'],dtype=torch.float32),
                'group':self.lookup.get(r[self.c['axis']],-1),'index':i}


def conditional_mmd(e,y,g,sigma):
    """Biased Gaussian MMD² to an equal-group mixture, averaged over classes.

    Computes squared RKHS distance via group sample weights. Both reference and
    subgroup evidence receive gradients. Float64 kernels avoid cancellation for
    nearly identical distributions. No cross-class terms or stop-gradient.
    """
    if e.ndim!=2 or y.shape!=(len(e),) or g.shape!=y.shape or not torch.isfinite(e).all():
        raise ValueError('Invalid evidence/label/group shape or finite values')
    if not ((y==0)|(y==1)).all() or g.dtype!=torch.long or (g<0).any():
        raise ValueError('Invalid labels/groups')
    if not math.isfinite(sigma) or sigma<=0:
        raise ValueError('Invalid kernel sigma')
    terms = []
    for label in (0,1):
        x = e[y==label].double(); ids = g[y==label]
        unique,counts = torch.unique(ids,return_counts=True)
        if len(unique)<2 or (counts<2).any():
            raise ValueError('Need >=2 groups and >=2 samples per class/group')
        w = (unique[:,None]==ids[None,:]).double()/counts[:,None]
        centered = w-w.mean(0,keepdim=True)
        kernel = torch.exp(-((x[:,None]-x[None,:])**2).sum(-1)/(2*sigma*sigma))
        terms.append(((centered@kernel)*centered).sum(1).mean())
    return torch.stack(terms).mean()


def soft_error_gap(z,y,g,threshold,temperature):
    if z.shape!=y.shape or g.shape!=y.shape or z.ndim!=1 or not torch.isfinite(z).all():
        raise ValueError('Invalid scores/labels/groups')
    if not ((y==0)|(y==1)).all() or g.dtype!=torch.long or (g<0).any():
        raise ValueError('Invalid labels/groups')
    if not math.isfinite(threshold) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid surrogate parameters')
    q = torch.sigmoid((z-threshold)/temperature)
    gaps = []
    for label in (0,1):
        ids = g[y==label]; values = q[y==label] if label==0 else 1-q[y==label]
        unique,counts = torch.unique(ids,return_counts=True)
        if len(unique)<2 or (counts<2).any():
            raise ValueError('Insufficient soft-rate support')
        rates = torch.stack([values[ids==group].mean() for group in unique])
        gaps.append(rates.max()-rates.min())
    return torch.stack(gaps).mean()


def objective(outputs,y,g,c):
    if len(outputs)!=len(CONDITIONS):
        raise ValueError('Need clean, jpeg, resize in declared order')
    bce, alignment, disparities = [],[],[]
    for o in outputs:
        z = o['logit']
        e = torch.stack([o['global_contribution'],o['local_contribution']],1)
        bce.append(F.binary_cross_entropy_with_logits(z,y))
        if c['alignment']=='joint':
            term = conditional_mmd(e,y,g,c['kernel_sigma'])
        elif c['alignment']=='score':
            term = conditional_mmd(z[:,None],y,g,c['kernel_sigma'])
        elif c['alignment']=='marginal':
            term = (conditional_mmd(e[:,:1],y,g,c['kernel_sigma'])+conditional_mmd(e[:,1:],y,g,c['kernel_sigma']))/2
        else:
            term = z.sum()*0
        alignment.append(term)
        disparities.append(soft_error_gap(z,y,g,c['train_threshold'],c['soft_temperature']))
    classification = torch.stack(bce).mean()
    align = torch.stack(alignment).mean()
    per_condition = torch.stack(disparities)
    condition = per_condition.max()
    total = classification+c['lambda_align']*align+c['lambda_condition']*condition
    return {'total':total,'classification':classification,'alignment':align,'condition':condition,
            'condition_disparities':per_condition}


def forward_conditions(model,images,device,chunk):
    n = len(images)
    flat = images.transpose(0,1).reshape(-1,*images.shape[2:])
    keys = ('logit','global_contribution','local_contribution')
    parts = {k:[] for k in keys}
    for start in range(0,len(flat),chunk):
        o = model(flat[start:start+chunk].to(device))
        for k in keys:
            parts[k].append(o[k])
    joined = {k:torch.cat(v) for k,v in parts.items()}
    return [{k:joined[k][j*n:(j+1)*n] for k in keys} for j in range(len(CONDITIONS))]


@torch.inference_mode()
def predict_conditions(model,rows,c,device):
    model.eval()
    result = {name:{k:[] for k in ('logit','global_contribution','local_contribution')} for name in CONDITIONS}
    loader = DataLoader(ConditionImages(rows,c),batch_size=c['eval_batch_size'],shuffle=False,
                        num_workers=c['workers'],generator=torch.Generator().manual_seed(c['seed']))
    for b in loader:
        outputs = forward_conditions(model,b['images'],device,c['forward_batch_size'])
        for name,o in zip(CONDITIONS,outputs):
            for k,v in o.items():
                result[name][k].extend(v.cpu().tolist())
    return {name:{k:np.asarray(v,dtype=float) for k,v in o.items()} for name,o in result.items()}


def report_conditions(rows,predictions,threshold,c):
    return {name:metrics(rows,p['logit'],threshold,c['min_support']) for name,p in predictions.items()}


def selection_key(reports,c,groups):
    if c['selection']=='validation_auc_only':
        return (-reports['clean']['overall']['auc'],)
    worst_gap = 0.; aucs = []
    for r in reports.values():
        axis = r['axes'][c['axis']]
        if not set(groups).issubset(axis['eligible_groups']):
            return None
        stats = [axis['groups'][g] for g in groups]
        auc = r['overall']['auc']; worst_auc = min(s['auc'] for s in stats)
        if auc<c['auc_floor'] or worst_auc<c['worst_group_auc_floor']:
            return None
        for metric in ('fpr','fnr'):
            values = [s[metric] for s in stats]
            worst_gap = max(worst_gap,max(values)-min(values))
        aucs.append(auc)
    return (worst_gap,-float(np.mean(aucs)))


def cancellation_report(rows,predictions,threshold,c):
    p = predictions['clean']; y = np.asarray([r['label'] for r in rows])
    gs = np.asarray([r[c['axis']] for r in rows])
    groups = sorted(set(gs)-{'unknown',''})
    records = []
    for label in (0,1):
        means = {g:np.array([p['global_contribution'][(y==label)&(gs==g)].mean(),
                             p['local_contribution'][(y==label)&(gs==g)].mean()])
                 for g in groups if ((y==label)&(gs==g)).sum()>=c['min_support']}
        if not means:
            continue
        reference = np.stack(list(means.values())).mean(0)
        for g,m in means.items():
            dg,dl = m-reference
            records.append({'label':label,'group':g,'n':int(((y==label)&(gs==g)).sum()),
                'global_mean':float(m[0]),'local_mean':float(m[1]),'global_shift':float(dg),
                'local_shift':float(dl),'cancellation_amount':float(max(0,abs(dg)+abs(dl)-abs(dg+dl)))})
    stress = {}
    for rho in (-.1,0.,.1):
        z = p['logit']+rho*(p['global_contribution']-p['local_contribution'])
        r = metrics(rows,z,threshold,c['min_support'])
        stress[str(rho)] = {'overall':r['overall'],'primary_axis':r['axes'][c['axis']]}
    return {'class_group_contributions':records,'shared_gain_stress':stress,
        'threshold_logit':threshold,'threshold_recalibrated':False,
        'interpretation':'Descriptive algebraic audit, not causal bias or corruption robustness. Actual condition metrics are reported separately; no audit gradients.',
        'uncertainty':'Point estimates only. Cluster bootstrap and multi-seed analysis remain required.'}


def check_device(device):
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested device unavailable; CUDA tests cannot be claimed')


def load_checkpoint(path,device):
    check_device(device)
    ck = torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('format')!=VERSION:
        raise ValueError('Not a fair-evidence checkpoint')
    c = ck['config']
    model = UpgradeCES(c['backbone'],hidden=c['hidden'],gate_hidden=c['gate_hidden'],pretrained=False,
                       encoder_policy=c['encoder_policy'],head_seed=c['head_seed'])
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck


def train(c,out,device,resume=False):
    check_device(device)
    rows,val,groups,cells,pre = preflight(c)
    source_hash = sha(__file__)
    out = Path(out)
    if not resume and out.exists():
        raise FileExistsError('Use a fresh output directory, or explicit --resume for an interrupted run')
    seed_all(c['seed']); torch.set_num_threads(4)
    model = UpgradeCES(c['backbone'],hidden=c['hidden'],gate_hidden=c['gate_hidden'],pretrained=False,
                       encoder_policy=c['encoder_policy'],head_seed=c['head_seed']).to(device)
    imported = import_encoder(model,c)
    optimizer = build_optimizer(model,c)
    start_epoch = 0; best = None; best_auc = -math.inf; best_epoch = None
    if resume:
        if (out/'DONE').exists():
            raise ValueError('Completed runs are immutable; choose a new output directory')
        ck = torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if ck.get('format')!=VERSION or ck['source_sha256']!=source_hash or ck['config']!=c:
            raise ValueError('Resume source/config/checkpoint mismatch')
        history = [json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in history] != list(range(1,ck['epoch']+1)):
            raise ValueError('Partial history/checkpoint mismatch; preserve artifacts for explicit recovery')
        model.load_state_dict(ck['model'],strict=True); optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch']; best = tuple(ck['best_key']) if ck['best_key'] is not None else None
        best_auc = ck['best_clean_auc']; best_epoch = ck['best_epoch']
        if best is not None and not (out/'best.pt').is_file():
            raise ValueError('Selected checkpoint missing')
        for filename in ('best.pt','best_auc.pt'):
            if (out/filename).exists():
                existing = torch.load(out/filename,map_location='cpu',weights_only=True)
                if existing['epoch']>start_epoch or existing['source_sha256']!=source_hash:
                    raise ValueError('Partial selected checkpoint; explicit recovery required')
    else:
        fresh_dir(out); write(out/'config.json',c)
        write(out/'provenance.json',{'format':VERSION,'source_sha256':source_hash,'preflight':pre,
            'input_hashes':c['input_hashes'],'parameters':model.counts(),'encoder_import':imported,
            'conditions':list(CONDITIONS),'bn_policy':'all running buffers frozen; selected affine parameters may train',
            'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
            'test_read':False,'generated_images_used':False,'selection':c['selection'],
            'image_content_hashes_verified':False,'note':'Manifest/checkpoint hashes verified; underlying image bytes depend on immutable cluster storage.'})
    dataset = ConditionImages(rows,c,groups)
    steps = pre['steps_per_epoch']; batch_size = pre['originals_per_step']
    started = time.perf_counter()
    for epoch in range(start_epoch,c['epochs']):
        # Epoch-local RNG makes deterministic epoch-boundary resume independent of validation work.
        seed_all(c['seed']+epoch)
        model.train()
        sampler = BalancedCells(cells,c['samples_per_cell'],steps,c['seed'],epoch)
        loader = DataLoader(dataset,batch_sampler=sampler,num_workers=c['workers'],
                            generator=torch.Generator().manual_seed(c['seed']+epoch))
        sums = {k:0. for k in ('total','classification','alignment','condition')}
        exposure = Counter(); disparity_sum = np.zeros(len(CONDITIONS))
        for step,b in enumerate(loader):
            outputs = forward_conditions(model,b['images'],device,c['forward_batch_size'])
            losses = objective(outputs,b['y'].to(device),b['group'].to(device),c)
            if not torch.isfinite(losses['total']):
                raise RuntimeError('Nonfinite objective')
            optimizer.zero_grad(set_to_none=True); losses['total'].backward()
            parameters = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(parameters,5.); optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach())
            disparity_sum += losses['condition_disparities'].detach().cpu().numpy()
            exposure.update(b['index'].tolist())
            if (step+1)%100==0:
                print(json.dumps({'epoch':epoch+1,'step':step+1,'steps':steps,
                    'mean_total_loss':sums['total']/(step+1)}),flush=True)
        predictions = predict_conditions(model,val,c,device)
        yy = np.asarray([r['label'] for r in val])
        calibration = calibrate(predictions['clean']['logit'][yy==0],c['target_fpr'])
        reports = report_conditions(val,predictions,calibration['threshold_logit'],c)
        key = selection_key(reports,c,groups)
        selected = key is not None and (best is None or key<best)
        if selected:
            best,best_epoch = key,epoch+1
        auc = reports['clean']['overall']['auc']; selected_auc = auc>best_auc
        if selected_auc:
            best_auc = auc
        audit = cancellation_report(val,predictions,calibration['threshold_logit'],c)
        record = {'epoch':epoch+1,'steps':steps,'losses':{k:v/steps for k,v in sums.items()},
            'condition_soft_gaps':dict(zip(CONDITIONS,(disparity_sum/steps).tolist())),
            'selection':c['selection'],'selection_key':key,'eligible_for_selection':key is not None,
            'selected_this_epoch':selected,'validation':reports,'calibration':calibration,
            'audit':audit,'original_draws':steps*batch_size,'image_forwards':steps*batch_size*len(CONDITIONS),
            'unique_originals_drawn':len(exposure),'source_exposure_min_drawn':min(exposure.values()),
            'source_exposure_max':max(exposure.values()),'elapsed_seconds':time.perf_counter()-started}
        checkpoint = {'format':VERSION,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'config':c,'epoch':epoch+1,'groups':groups,'source_sha256':source_hash,'input_hashes':c['input_hashes'],
            'validation':reports,'calibration':calibration,'best_key':best,'best_epoch':best_epoch,
            'best_clean_auc':best_auc,'selection_key':key,'checkpoint_role':'last'}
        if selected:
            atomic_save({**checkpoint,'checkpoint_role':'primary_selection'},out/'best.pt')
        if selected_auc:
            atomic_save({**checkpoint,'checkpoint_role':'clean_auc_comparator'},out/'best_auc.pt')
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        atomic_save(checkpoint,out/'last.pt')
        print(json.dumps({'epoch':epoch+1,'clean_validation_auc':auc,'selection_key':key,
                         'best_epoch':best_epoch,'losses':record['losses']}),flush=True)
    summary = {'state':'COMPLETED_SELECTED' if best is not None else 'COMPLETED_NO_ELIGIBLE_CHECKPOINT',
        'best_epoch':best_epoch,'selection':c['selection'],'best_key':best,
        'best_clean_auc':best_auc,'test_evaluated':False,
        'checkpoint':str(out/'best.pt') if best is not None else None,
        'note':'No automatic fallback if utility floors fail; best_auc.pt is a separately labeled comparator.'}
    write(out/'summary.json',summary)
    (out/'DONE').write_text('Training and validation complete; no test evaluation.\n')
    if best is not None:
        verify_selected(out,device)
    return summary


def dry_run(c,out,device):
    """Exercise one real configured batch and backward pass; no optimizer update."""
    check_device(device)
    rows,_,groups,cells,pre = preflight(c)
    target = fresh_dir(out)
    seed_all(c['seed']); torch.set_num_threads(4)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    model = UpgradeCES(c['backbone'],hidden=c['hidden'],gate_hidden=c['gate_hidden'],pretrained=False,
                       encoder_policy=c['encoder_policy'],head_seed=c['head_seed']).to(device)
    import_encoder(model,c); model.train()
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    loader = DataLoader(ConditionImages(rows,c,groups),
        batch_sampler=BalancedCells(cells,c['samples_per_cell'],1,c['seed'],0),num_workers=c['workers'],
        generator=torch.Generator().manual_seed(c['seed']))
    b = next(iter(loader))
    started = time.perf_counter()
    losses = objective(forward_conditions(model,b['images'],device,c['forward_batch_size']),
                       b['y'].to(device),b['group'].to(device),c)
    if not torch.isfinite(losses['total']): raise RuntimeError('Nonfinite dry-run loss')
    losses['total'].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not all(g is not None and torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Missing/nonfinite trainable gradient')
    if not any(torch.count_nonzero(g) for g in grads): raise RuntimeError('All gradients zero')
    if not all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items()):
        raise RuntimeError('Dry run changed parameters or buffers')
    if device=='cuda': torch.cuda.synchronize()
    result = {'status':'PASS','device':device,'source_sha256':sha(__file__),
        'config':c,'preflight':pre,'parameters':model.counts(),
        'losses':{k:float(losses[k].detach()) for k in ('total','classification','alignment','condition')},
        'backward_seconds':time.perf_counter()-started,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'optimizer_steps':0,'parameters_and_buffers_unchanged':True,
        'note':'One configured batch only; does not validate fairness or reserve GPU resources.'}
    write(target/'RESULT.json',result)
    return result


def verify_selected(out,device):
    """Independent reload and validation replay of the actual selected checkpoint."""
    out = Path(out); model,ck = load_checkpoint(out/'best.pt',device); c = ck['config']
    if sha(c['val'])!=ck['input_hashes']['val']:
        raise ValueError('Validation manifest changed before verification')
    rows = read_rows(c['val']); predictions = predict_conditions(model,rows,c,device)
    report = report_conditions(rows,predictions,ck['calibration']['threshold_logit'],c)
    y = np.asarray([r['label'] for r in rows])
    replay_cal = calibrate(predictions['clean']['logit'][y==0],c['target_fpr'])
    threshold_match = math.isclose(replay_cal['threshold_logit'],ck['calibration']['threshold_logit'],rel_tol=1e-6,abs_tol=1e-6)
    differences = []
    for name in CONDITIONS:
        for key in ('auc','fpr','fnr'):
            differences.append(abs(report[name]['overall'][key]-ck['validation'][name]['overall'][key]))
        for group,stats in ck['validation'][name]['axes'][c['axis']]['groups'].items():
            now = report[name]['axes'][c['axis']]['groups'][group]
            for key in ('auc','fpr','fnr'):
                if stats[key] is not None:
                    differences.append(abs(now[key]-stats[key]))
    matched = threshold_match and max(differences)<1e-5
    receipt = {'status':'PASS' if matched else 'FAIL','checkpoint_sha256':sha(out/'best.pt'),
        'source_sha256':sha(__file__),'device':device,'selected_epoch':ck['epoch'],
        'validation_forward_independently_replayed':True,'max_metric_difference':max(differences),
        'calibration_threshold_matches':threshold_match,'test_read':False,
        'meaning':'Software/checkpoint consistency only; not fairness efficacy or novelty verification.'}
    write(out/('RUN_VERIFIED.json' if matched else 'VERIFICATION_FAILED.json'),receipt)
    if not matched:
        raise RuntimeError('Selected checkpoint validation replay differs; inspect verification receipt')
    return receipt


def evaluate(checkpoint,manifest,out,device,split='test'):
    check_device(device)
    if Path(out).exists():
        raise FileExistsError('Preserve prior evaluation')
    model,ck = load_checkpoint(checkpoint,device); c = ck['config']
    manifest_hash = sha(manifest)
    if split=='test' and manifest_hash in (ck['input_hashes']['train'],ck['input_hashes']['val']):
        raise ValueError('TRAIN/validation manifest cannot be labeled test')
    if split=='validation' and manifest_hash!=ck['input_hashes']['val']:
        raise ValueError('Not the original validation manifest')
    if split=='train' and manifest_hash!=ck['input_hashes']['train']:
        raise ValueError('Not the original TRAIN manifest')
    if split not in ('test','validation','train'):
        raise ValueError('Unknown evaluation role')
    rows = read_rows(manifest); predictions = predict_conditions(model,rows,c,device)
    threshold = ck['calibration']['threshold_logit']
    report = {'format':VERSION,'split':split,'checkpoint_sha256':sha(checkpoint),
        'manifest_sha256':manifest_hash,'checkpoint_role':ck['checkpoint_role'],'checkpoint_epoch':ck['epoch'],
        'threshold_source':'selected checkpoint clean validation only','calibration':ck['calibration'],
        'calibrated':report_conditions(rows,predictions,threshold,c),
        'fixed_probability_0_5':report_conditions(rows,predictions,0.,c),
        'audit':cancellation_report(rows,predictions,threshold,c),
        'evaluation_source_sha256':sha(__file__),'training_source_sha256':ck['source_sha256'],
        'caution':'Point estimates; test membership/provenance must be reviewed externally. No tuning or threshold recalibration.'}
    target = fresh_dir(out); write(target/'metrics.json',report)
    (target/'DONE').write_text('Evaluation complete with frozen threshold.\n')
    return report


@torch.inference_mode()
def export_maps(checkpoint,images,out,device):
    model,ck = load_checkpoint(checkpoint,device); c = ck['config']; target = fresh_dir(out)
    scores = []
    for i,path in enumerate(images):
        with Image.open(path) as im:
            original = im.convert('RGB'); inp = image_conditions(original,c)[0]
        r = model(tensor_image(inp,c['backbone'])[None].to(device))
        raw = r['local_map'][0].cpu().numpy(); gate = r['patch_gate'][0].cpu().numpy()
        torch.testing.assert_close(r['logit'],r['global_contribution']+r['local_map'].mean((1,2))+r['fusion_bias'])
        upsample = F.interpolate(r['local_map'][:,None],size=(original.height,original.width),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        scale = max(float(np.max(abs(upsample))),1e-12)
        heat = Image.fromarray(diverging(upsample,scale)); name = f'{i:04d}'
        heat.save(target/(name+'_signed_local.png')); Image.blend(original,heat,.45).save(target/(name+'_overlay.png'))
        np.savez_compressed(target/(name+'_raw.npz'),local_map=raw,patch_gate=gate)
        scores.append({'source_sha256':sha(path),'logit':float(r['logit'][0]),
            'global_contribution':float(r['global_contribution'][0]),'local_contribution':float(r['local_contribution'][0]),
            'fusion_bias':float(r['fusion_bias']),'image_gate':float(r['image_gate'][0]),
            'display_scale':scale,'threshold_logit':ck['calibration']['threshold_logit']})
    write(target/'scores.json',scores)
    (target/'README.txt').write_text('Red supports fake, blue supports real: signed LOCAL contribution only. Global and bias separate. Gates are learned weights, not reliability. Maps are not forgery masks. Per-image display scale; compare raw maps on a common scale. Keep face images outside public Git.\n')
    return scores


def smoke(device,out):
    """Self-contained numerical, gradient, data, train/reload/resume/evaluate checks."""
    check_device(device); target = fresh_dir(out); seed_all(17); torch.set_num_threads(2)
    checks = []
    def passed(name):
        checks.append(name); print('SMOKE_PASS',name,flush=True)
    # Independent kernel-mixture reference with unequal subgroup sample sizes.
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(14,2)); yy = np.array([0]*5+[1]*9)
    gg = np.array([0]*2+[1]*3+[0]*4+[1]*5)
    expected = 0.
    for label in (0,1):
        a = raw[yy==label]; ids = gg[yy==label]
        kernel = np.exp(-((a[:,None]-a[None,:])**2).sum(-1)/32.)
        reference = np.array([1/(2*np.sum(ids==g)) for g in ids])
        for g in (0,1):
            weights = (ids==g)/np.sum(ids==g)-reference
            expected += float(weights@kernel@weights)/4
    e = torch.tensor(raw,dtype=torch.float64,device=device,requires_grad=True)
    y = torch.tensor(yy,device=device); g = torch.tensor(gg,device=device)
    actual = conditional_mmd(e,y,g,4.)
    assert abs(float(actual.detach())-expected)<1e-12
    assert torch.autograd.gradcheck(lambda v:conditional_mmd(v,y,g,4.),(e,))
    passed('mmd_matches_independent_equal_group_mixture_and_finite_difference_gradient')
    shifted = e+100*y[:,None]
    torch.testing.assert_close(actual,conditional_mmd(shifted,y,g,4.))
    with torch.no_grad():
        same = torch.tensor([[-2.,-1.],[-1.,-2.]]*2+[[1.,2.],[2.,1.]]*2,device=device)
        y2 = torch.tensor([0]*4+[1]*4,device=device)
        g2 = torch.tensor([0,0,1,1]*2,device=device)
        assert abs(float(conditional_mmd(same,y2,g2,4.)))<1e-12
    passed('classes_never_aligned_to_each_other')
    offset = torch.tensor(3.,dtype=torch.float64,device=device,requires_grad=True)
    s = torch.tensor([-2.,-1.,-2.,-1.,1.,2.,1.,2.],dtype=torch.float64,device=device)
    sign = torch.tensor([1.,1.,-1.,-1.]*2,device=device)
    cancel = torch.stack([s/2+offset*sign,s/2-offset*sign],1)
    final_loss = conditional_mmd(cancel.sum(1,keepdim=True),y2,g2,4.)
    joint = conditional_mmd(cancel,y2,g2,4.)
    assert abs(float(final_loss.detach()))<1e-12 and float(joint.detach())>.01
    grad = torch.autograd.grad(joint,offset)[0]
    assert abs(float(grad))>1e-5
    passed('joint_alignment_detects_constructed_score_cancellation')
    z = torch.tensor([-2.,-1.,0.,1.,-1.,0.,1.,2.],device=device,requires_grad=True)
    gap = soft_error_gap(z,y2,g2,0.,1.)
    q = 1/(1+np.exp(-z.detach().cpu().numpy()))
    reference = ((q[2:4].mean()-q[:2].mean())+((1-q[4:6]).mean()-(1-q[6:]).mean()))/2
    assert abs(float(gap.detach())-reference)<1e-6
    gap.backward(); assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    outputs = [{'logit':(z.detach()+k).requires_grad_(),
                'global_contribution':(z.detach()*.7+k).requires_grad_(),
                'local_contribution':(z.detach()*.3).requires_grad_()} for k in (0.,.1,-.1)]
    for arm in ('bce','score','marginal','joint','full'):
        c = default_config(arm)
        loss = objective(outputs,y2.float(),g2,c)
        torch.testing.assert_close(loss['condition'],loss['condition_disparities'].max())
        loss['total'].backward()
        assert all(torch.isfinite(o['logit'].grad).all() for o in outputs)
        for o in outputs:
            for value in o.values(): value.grad=None
    passed('all_five_objectives_and_worst_condition_soft_error_gradients')
    cells = {(0,0):[0,1,2],(0,1):[3,4,5],(1,0):[6,7,8],(1,1):[9,10,11]}
    a = list(BalancedCells(cells,2,3,17,0)); b = list(BalancedCells(cells,2,3,17,0))
    assert a==b and all(len(set(batch))==8 for batch in a)
    for batch in a:
        assert all(len(set(batch)&set(ix))==2 for ix in cells.values())
    pixels = rng.integers(0,256,(75,75,3),dtype=np.uint8)
    c = {**default_config(),'size':75}
    conditions = image_conditions(Image.fromarray(pixels),c)
    assert np.array_equal(np.asarray(conditions[0]),pixels)
    assert all(im.size==(75,75) for im in conditions)
    assert not np.array_equal(np.asarray(conditions[0]),np.asarray(conditions[1]))
    assert not np.array_equal(np.asarray(conditions[0]),np.asarray(conditions[2]))
    assert calibrate([.7]*4,.25)['achieved_fp']==0
    passed('matched_distinct_cell_sampling_independent_conditions_and_tied_threshold')
    # Full default-width architecture at production resolution, no downloaded weights.
    model = UpgradeCES(pretrained=False,head_seed=17).to(device).train()
    before = {k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    r = model(torch.randn(2,3,224,224,device=device))
    torch.testing.assert_close(r['logit'],r['global_contribution']+r['local_map'].mean((1,2))+r['fusion_bias'])
    F.binary_cross_entropy_with_logits(r['logit'],torch.tensor([0.,1.],device=device)).backward()
    for branch in ('global_head','patch_head','image_gate','patch_gate'):
        grads = [p.grad for p in getattr(model,branch).parameters()]
        assert all(v is not None and torch.isfinite(v).all() for v in grads)
        assert sum(float(v.abs().sum()) for v in grads)>0,branch
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.requires_grad and p.grad is not None)>0
    opt = build_optimizer(model,default_config()); opt.step()
    changed = []
    for k,v in model.encoder.state_dict().items():
        if not torch.equal(v.cpu(),before[k]): changed.append(k)
        if any(tag in k for tag in ('running_mean','running_var','num_batches_tracked')):
            assert torch.equal(v.cpu(),before[k]),k
    assert changed and all(any(k.startswith(n+'.') for n in model.selected_names) for k in changed)
    counts = model.counts()
    assert counts=={'total':21594159,'trainable':7575591},counts
    del model,before,opt,r
    if device=='cuda': torch.cuda.empty_cache()
    passed('224px_full_width_xception_gradients_frozen_bn_exact_decomposition_and_parameter_counts')
    with tempfile.TemporaryDirectory(prefix='cesfl_fair_evidence_smoke_') as directory:
        root = Path(directory).resolve(); paths = {}
        for split in ('train','val','test'):
            rows = []
            for i in range(8):
                path = root/f'{split}_{i}.png'
                Image.fromarray(rng.integers(0,256,(75,75,3),dtype=np.uint8)).save(path)
                rows.append({'image':str(path),'label':i%2,'identity_ids':f'{split}{i}',
                    'source_ids':f'{split}{i}','gender_race':str(i//4),'family':'synthetic','cluster':f'{split}{i}'})
            paths[split] = str(root/f'{split}.csv')
            with Path(paths[split]).open('w',newline='') as f:
                w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        config = {**default_config(),'train':paths['train'],'val':paths['val'],
            'protocol_overlap_file':None,'expected_groups':2,'min_train_cell':2,'min_support':2,
            'hidden':16,'gate_hidden':8,'size':75,'epochs':2,'steps_per_epoch':1,'workers':0,
            'forward_batch_size':4,'eval_batch_size':4,'auc_floor':0.,'worst_group_auc_floor':0.}
        source = DiagramCES(pretrained=False,hidden=16,gate_hidden=8,head_seed=17)
        ckpath = root/'synthetic_pretrain.pt'
        torch.save({'format':PRETRAIN_FORMAT,'model':source.state_dict(),'epoch':1,
            'config':{**config,'stage':'pretrain','encoder_policy':'trainable','mode':'bce'},
            'training_inputs':{'train_sha256':sha(paths['train']),'val_sha256':sha(paths['val'])}},ckpath)
        del source
        config.update(encoder_checkpoint=str(ckpath),encoder_checkpoint_sha256=sha(ckpath))
        config['input_hashes'] = {k:sha(config[k]) for k in ('train','val','encoder_checkpoint')}
        validate_config(config); preflight(config)
        for change in ({'unexpected':True},{'lambda_align':float('nan')},{'samples_per_cell':1},
                       {'jpeg_quality':20},{'axis':'made_up'},{'resize_scale':0}):
            try: validate_config({**config,**change})
            except ValueError: pass
            else: raise AssertionError('Invalid configuration accepted')
        try: preflight({**config,'input_hashes':{**config['input_hashes'],'train':'0'*64}})
        except ValueError: pass
        else: raise AssertionError('Changed manifest accepted')
        passed('strict_config_provenance_and_pretraining_transfer_preflight')
        dry = dry_run(config,root/'dry_run',device)
        assert dry['optimizer_steps']==0 and dry['parameters_and_buffers_unchanged']
        passed('configured_batch_backward_dry_run_without_parameter_or_buffer_updates')
        full = train(config,root/'full',device)
        assert full['state']=='COMPLETED_SELECTED'
        assert (root/'full/RUN_VERIFIED.json').is_file()
        model,ck = load_checkpoint(root/'full/best.pt',device)
        assert ck['config']['lambda_condition']>0
        noeligible = selection_key(ck['validation'],{**config,'auc_floor':.999999},ck['groups'])
        # Explicit collapse case, independent of random fixture AUC.
        collapsed = json.loads(json.dumps(ck['validation']))
        collapsed['jpeg']['overall']['auc'] = 0.
        assert selection_key(collapsed,{**config,'auc_floor':.9},ck['groups']) is None
        passed('full_training_save_reload_utility_gate_and_independent_validation_replay')
        # Simulate interruption immediately after a complete epoch-boundary checkpoint.
        original_save = globals()['atomic_save']
        def interrupt_after_first(value,path):
            original_save(value,path)
            if Path(path).name=='last.pt' and value['epoch']==1:
                raise InterruptedError('SYNTHETIC controlled epoch-boundary interruption')
        globals()['atomic_save'] = interrupt_after_first
        try:
            try: train(config,root/'resumed',device)
            except InterruptedError: pass
            else: raise AssertionError('Interruption did not occur')
        finally:
            globals()['atomic_save'] = original_save
        train(config,root/'resumed',device,resume=True)
        one = torch.load(root/'full/last.pt',map_location='cpu',weights_only=True)
        two = torch.load(root/'resumed/last.pt',map_location='cpu',weights_only=True)
        for k,v in one['model'].items():
            torch.testing.assert_close(v,two['model'][k],rtol=0,atol=0)
        del one,two,model,ck
        bce_config = {**config,'arm':'bce','alignment':'none','lambda_align':0.,'lambda_condition':0.,'epochs':1}
        train(bce_config,root/'bce',device)
        passed('deterministic_epoch_boundary_resume_and_matched_bce_training')
        first = evaluate(root/'full/best.pt',paths['test'],root/'eval1',device)
        with Path(paths['test']).open() as f: rows = list(csv.DictReader(f))
        for row in rows: row['label']=str(1-int(row['label']))
        flipped = root/'test_flipped.csv'
        with flipped.open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        second = evaluate(root/'full/best.pt',flipped,root/'eval2',device)
        assert first['calibration']==second['calibration']
        assert first['audit']['threshold_logit']==second['audit']['threshold_logit']
        try: evaluate(root/'full/best.pt',paths['val'],root/'bad_test',device)
        except ValueError: pass
        else: raise AssertionError('Validation mislabeled test')
        exported = export_maps(root/'full/best.pt',[rows[0]['image']],root/'maps',device)
        assert abs(exported[0]['logit']-exported[0]['global_contribution']-exported[0]['local_contribution']-exported[0]['fusion_bias'])<1e-5
        passed('evaluation_label_flip_keeps_threshold_audit_and_map_export')
    result = {'status':'PASS','device':device,'checks':checks,'check_count':len(checks),
        'parameters':counts,'source_sha256':sha(__file__),'torch':str(torch.__version__),
        'python':sys.version.split()[0],'data':'synthetic software fixtures only; temporary images/checkpoints removed',
        'production_cluster_data_tested':False,'fairness_improvement_verified':False,
        'novelty_established':False,'cuda_tested':device=='cuda'}
    write(target/'RESULT.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('init-config',help='Write pinned cluster defaults; no training')
    p.add_argument('--arm',choices=['bce','score','marginal','joint','full'],default='full'); p.add_argument('--out',required=True)
    p = sub.add_parser('preflight',help='Read/hash manifests and pretraining metadata; no training')
    p.add_argument('--config',required=True)
    p = sub.add_parser('dry-run',help='One configured real-data batch/backward, no optimizer steps')
    p.add_argument('--config',required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p = sub.add_parser('smoke',help='Synthetic end-to-end software tests, no external weights/data')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--out',required=True)
    p = sub.add_parser('train',help='Train all declared conditions; validate; independently verify selected checkpoint')
    p.add_argument('--config',required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--resume',action='store_true')
    for command in ('evaluate','audit'):
        p = sub.add_parser(command)
        p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
        p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        p.add_argument('--split',choices=['train','validation'] if command=='audit' else ['test','validation'],
                       default='validation' if command=='audit' else 'test')
    p = sub.add_parser('export',help='Export signed local maps and exact contribution scores')
    p.add_argument('--checkpoint',required=True); p.add_argument('--images',nargs='+',required=True)
    p.add_argument('--out',required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a = parser.parse_args()
    if a.command=='init-config':
        c = default_config(a.arm); validate_config(c)
        path = Path(a.out); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f: f.write(json.dumps(c,indent=2)+'\n')
        print('CONFIG_WRITTEN',path,'NO_TRAINING_STARTED')
    elif a.command=='preflight':
        print(json.dumps(preflight(json.loads(Path(a.config).read_text()))[-1],indent=2))
    elif a.command=='smoke':
        smoke(a.device,a.out)
    elif a.command=='dry-run':
        print(json.dumps(dry_run(json.loads(Path(a.config).read_text()),a.out,a.device),indent=2))
    elif a.command=='train':
        result = train(json.loads(Path(a.config).read_text()),a.out,a.device,a.resume)
        print(json.dumps(result,indent=2))
        if result['state']=='COMPLETED_NO_ELIGIBLE_CHECKPOINT': sys.exit(2)
    elif a.command in ('evaluate','audit'):
        result = evaluate(a.checkpoint,a.manifest,a.out,a.device,a.split)
        print(json.dumps({'split':result['split'],'conditions':list(result['calibrated']),
                         'threshold_source':result['threshold_source'],'out':a.out}))
    else:
        export_maps(a.checkpoint,a.images,a.out,a.device)


if __name__=='__main__':
    main()
