#!/usr/bin/env python3
"""CES-FL matched 50+50 counterfactual experiment. Standalone, 1 November 2026.
No downloads or implicit GPU jobs. Generation and approval are separate stages.
Every TRAIN original must have its own approved age/gender-presentation/skin edit
and no-change control. No edits are fabricated or transferred between people.
E-box: z = global_contribution + mean(local_map) + fusion_bias.
Gates are learned weights, maps are signed contributions, not forgery masks.
Embedded architecture/utilities attributed to CES-FL v0.2.1 SHA256
0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de;
Xception originates from the project's vendored Ding utils/xception.py.
Requires Python >=3.10, torch, numpy, Pillow, scikit-learn. Qwen generation
additionally needs diffusers/transformers/accelerate and local model weights.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
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
from torch.utils.data import Dataset, DataLoader
VERSION='CESFL_CF5050_v1'
DEFAULT_XCEPTION=None
AXES=('gender','race','age_bin','skin_bin','gender_race','ids8')
ATTRIBUTES=('age','gender','skin')
DIRECTIONS={'age':('younger','older'),'gender':('gender_feminine','gender_masculine'),'skin':('skin_lighter','skin_darker')}

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

def flag(value):
    s=str(value).strip().lower()
    if s not in ('','0','1','false','true','no','yes'):raise ValueError(f'Invalid boolean flag: {value}')
    return s in ('1','true','yes')

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

class PairImages(Dataset):
    def __init__(self,rows,pairs=None,seed=17,epoch=0,mode='bce',size=224,backbone='xception'):
        self.rows,self.pairs,self.seed,self.epoch,self.mode,self.size,self.backbone=rows,pairs or {},seed,epoch,mode,size,backbone
    def __len__(self):return len(self.rows)
    def load(self,path):
        with Image.open(path) as im:a=np.asarray(im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32).copy()/255.
        x=torch.from_numpy(a).permute(2,0,1)
        if self.backbone=='xception':return (x-.5)/.5
        return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    def __getitem__(self,i):
        r=self.rows[i];demo=[];forensic=[]
        for attr,choices in sorted(self.pairs.get(r['image'],{}).items()):
            if attr=='forensic' and self.mode not in ('sensitivity','combined'):continue
            if attr!='forensic' and self.mode not in ('invariance','combined'):continue
            offset=int(hashlib.sha256(f'{self.seed}:{r["image"]}:{attr}'.encode()).hexdigest()[:16],16)
            p=choices[(offset+self.epoch)%len(choices)]
            (forensic if attr=='forensic' else demo).append((self.load(p['counterfactual_image']),p['pair_id'],attr,p['spatial_validated']))
        return self.load(r['image']),r['label'],demo,forensic,i

def collate(items):
    result={'x':torch.stack([r[0] for r in items]),'y':torch.tensor([r[1] for r in items],dtype=torch.float32),'rows':[r[4] for r in items]}
    for key,pos in [('demo',2),('forensic',3)]:
        pairs=[(j,p) for j,r in enumerate(items) for p in r[pos]]
        result[key]=torch.stack([p[0] for _,p in pairs]) if pairs else None
        result[key+'_index']=torch.tensor([j for j,_ in pairs],dtype=torch.long)
        result[key+'_ids']=[p[1] for _,p in pairs];result[key+'_attributes']=[p[2] for _,p in pairs]
        result[key+'_spatial']=torch.tensor([p[3] for _,p in pairs],dtype=torch.bool)
    return result

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

@torch.inference_mode()
def predict(model,rows,device,batch_size=16,workers=0,size=224):
    model.eval();out=[]
    for b in DataLoader(PairImages(rows,size=size,backbone=model.backbone),batch_size=batch_size,collate_fn=collate,num_workers=workers):
        out.extend(model(b['x'].to(device))['logit'].cpu().tolist())
    return np.asarray(out)

def build_optimizer(model,c):
    encoder=[p for p in model.encoder.parameters() if p.requires_grad]
    other=[p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    groups=[]
    if encoder:groups.append({'params':encoder,'lr':c.get('encoder_lr',c['lr']),'name':'encoder'})
    if other:groups.append({'params':other,'lr':c['lr'],'name':'heads_gates_fusion'})
    if not groups:raise ValueError('No trainable parameters')
    if not all(math.isfinite(g['lr']) and g['lr']>0 for g in groups):raise ValueError('Invalid learning rate')
    return torch.optim.AdamW(groups,lr=c['lr'],weight_decay=c['weight_decay'])

def diverging(a,scale):
    t=np.clip(np.asarray(a,dtype=float)/max(scale,1e-12),-1,1);out=np.ones((*t.shape,3),float)
    out[...,0]=1-np.maximum(-t,0);out[...,1]=1-abs(t);out[...,2]=1-np.maximum(t,0)
    return (out*255).astype(np.uint8)


def atomic_json(path,value):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_checkpoint(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)


def fresh(path):
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=False);return path


def config_default():
    root='/home/sachin.chaudhary/ces_fl_pipeline'
    return dict(seed=17,head_seed=17,backbone='xception',hidden=128,gate_hidden=64,size=224,
        pretrain_epochs=50,downstream_epochs=50,batch_size=16,workers=4,balance_axis='ids8',
        lr=.001,pretrain_encoder_lr=.0001,encoder_lr=.00001,weight_decay=.0001,
        lambda_d=1.,target_fpr=.1,min_support=20,selection='validation_auc_only',
        train=root+'/manifests/ding_ffpp/train.csv',val=root+'/manifests/ding_ffpp/val.csv',
        protocol_overlap_file=root+'/manifests/ding_ffpp/protocol_source_overlap.json',
        weights='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data/xception-b5690688.pth',
        input_hashes={'train':'6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657',
            'val':'52f4844cffe5a03adb08cecde8829244bb50e05eeead36f6d659599f1c24f5b1',
            'protocol_overlap_file':'793db56c30c85dd654c6d68745afc926f4b1f8dc8bec749db01b07b3aa306c97',
            'weights':'43020ad2812c333d9f8e202b9fef94e579d07addb6582d69a1f68bbaec21aa90'},
        model_snapshot='/home/sachin.chaudhary/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9',
        generation_steps=40,generation_cfg=4.,generation_area=1024**2,
        pair_bank=None,pair_bank_sha256=None,pretrain_checkpoint=None,pretrain_checkpoint_sha256=None)


def validate_config(c,synthetic=False):
    if set(c)!=set(config_default()):raise ValueError('Configuration keys differ from schema')
    for k in ('seed','head_seed','workers'):
        if type(c[k]) is not int or c[k]<0:raise ValueError('Invalid '+k)
    for k in ('hidden','gate_hidden','batch_size','min_support','pretrain_epochs','downstream_epochs','generation_steps','generation_area'):
        if type(c[k]) is not int or c[k]<1:raise ValueError('Invalid '+k)
    if c['batch_size']<2 or type(c['size']) is not int or c['size']<75:raise ValueError('Invalid batch/image size')
    if not synthetic and (c['pretrain_epochs']!=50 or c['downstream_epochs']!=50):
        raise ValueError('Production protocol requires exactly 50+50 epochs')
    if c['backbone']!='xception':raise ValueError('This reviewed launch is Xception; ResNet needs a separate matched protocol')
    if c['balance_axis'] not in AXES or c['selection']!='validation_auc_only':raise ValueError('Invalid protocol')
    for k in ('lr','pretrain_encoder_lr','encoder_lr','lambda_d','generation_cfg'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k]<=0:raise ValueError('Invalid '+k)
    if not math.isfinite(c['weight_decay']) or c['weight_decay']<0:raise ValueError('Invalid decay')
    if not math.isfinite(c['target_fpr']) or not 0<=c['target_fpr']<1:raise ValueError('Invalid target FPR')
    keys={'train','val','weights'}|({'protocol_overlap_file'} if c['protocol_overlap_file'] else set())
    if set(c['input_hashes'])!=keys:raise ValueError('Exact external input hash keys required')
    for k in keys:
        if sha(c[k])!=c['input_hashes'][k]:raise ValueError('Input changed: '+k)
    for k in ('pair_bank','pretrain_checkpoint'):
        if bool(c[k])!=bool(c[k+'_sha256']):raise ValueError('Missing bound hash: '+k)
        if c[k] and sha(c[k])!=c[k+'_sha256']:raise ValueError('Bound input changed: '+k)


def data_preflight(c,synthetic=False):
    validate_config(c,synthetic)
    rows,val=read_rows(c['train']),read_rows(c['val'])
    split=check_splits(rows,val,c['protocol_overlap_file'])
    if any(r[c['balance_axis']].strip().lower() in ('unknown','','none','nan') for r in rows):
        raise ValueError('Unknown balancing groups; do not silently relabel')
    counts=Counter((r['label'],r[c['balance_axis']]) for r in rows)
    groups={r[c['balance_axis']] for r in rows}
    if any((y,g) not in counts for y in (0,1) for g in groups):raise ValueError('Both classes required per balancing group')
    weights=[len(rows)/(len(counts)*counts[(r['label'],r[c['balance_axis']])]) for r in rows]
    return rows,val,weights,split


PRESERVE=' Preserve the same identity, pose, expression, gaze, framing, background, lighting, clothing and unrelated details. Preserve image texture and existing manipulation artifacts. Do not beautify, repair, denoise, sharpen or replace the person.'
PROMPTS={
    'nochange':'Keep the supplied image unchanged, including apparent age, gender presentation, skin tone and every visible detail.'+PRESERVE,
    'younger':'Make this same person appear younger as an adult, changing only age-related appearance. Retain natural skin texture and recognizable facial structure.'+PRESERVE,
    'older':'Make this same person appear noticeably older, with natural age-related facial changes only.'+PRESERVE,
    'gender_feminine':'Make only the gender-related visual presentation of this same person subtly more feminine. Preserve bone structure, apparent age, skin tone, hair and clothing.'+PRESERVE,
    'gender_masculine':'Make only the gender-related visual presentation of this same person subtly more masculine. Preserve bone structure, apparent age, skin tone, hair and clothing.'+PRESERVE,
    'skin_lighter':'Make only visible skin one natural shade lighter. Preserve ethnicity-related facial features, apparent age and gender presentation.'+PRESERVE,
    'skin_darker':'Make only visible skin one natural shade darker. Preserve ethnicity-related facial features, apparent age and gender presentation.'+PRESERVE}


def source_id(path):return hashlib.sha256(str(path).encode()).hexdigest()[:24]


def plan_generation(c,out,synthetic=False):
    rows,_,_,_=data_preflight(c,synthetic);out=fresh(out)
    sources=[{'source_id':source_id(r['image']),'original_image':r['image'],'original_sha256':sha(r['image']),
        'label':r['label'],'identity_ids':r['identity_ids'],'source_ids':r['source_ids']} for r in rows]
    if len({r['source_id'] for r in sources})!=len(sources):raise ValueError('Source ID collision')
    write(out/'sources.json',sources)
    plan={'format':VERSION,'source_sha256':sha(__file__),'config':c,'sources_sha256':sha(out/'sources.json'),
        'prompts':PROMPTS,'source_count':len(sources),'outputs_per_source':7,'generated_outputs':7*len(sources),
        'resize_controls':len(sources),'independent_original_inputs':True,'auto_approval':False,
        'model_binary_hashes_verified':False,'synthetic_fixture':synthetic}
    write(out/'plan.json',plan);print(json.dumps({k:plan[k] for k in ('source_count','generated_outputs','auto_approval')}),flush=True)
    return out


def load_plan(path,synthetic=False):
    path=Path(path).resolve();plan=json.loads((path/'plan.json').read_text())
    if plan['format']!=VERSION or plan['source_sha256']!=sha(__file__):raise ValueError('Generation source/version changed')
    if plan['synthetic_fixture'] and not synthetic:raise ValueError('Synthetic generation plan cannot be used in production')
    if sha(path/'sources.json')!=plan['sources_sha256'] or plan['prompts']!=PROMPTS:raise ValueError('Generation plan changed')
    sources=json.loads((path/'sources.json').read_text())
    if len(sources)!=plan['source_count'] or len({r['source_id'] for r in sources})!=len(sources):raise ValueError('Invalid plan sources')
    return path,plan,sources


def candidate_seed(seed,sid):
    # Same per-source seed across all conditions; source itself always the original.
    return int(hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()[:15],16)


def generate(plan_dir,shard_index=0,num_shards=1,max_sources=None,synthetic_generator=None):
    root,plan,sources=load_plan(plan_dir,synthetic_generator is not None);c=plan['config']
    if num_shards<1 or not 0<=shard_index<num_shards or (max_sources is not None and max_sources<1):raise ValueError('Invalid shard/limit')
    # Immutable source partition, one writer per source. Locks prevent overlap.
    selected=[s for i,s in enumerate(sources) if i%num_shards==shard_index]
    if max_sources is not None:
        # Timing/visual pilot includes both classes when this shard contains both.
        by_label={y:[s for s in selected if s['label']==y] for y in (0,1)}
        preview=[];positions={0:0,1:0}
        while len(preview)<min(max_sources,len(selected)):
            for y in (0,1):
                if positions[y]<len(by_label[y]) and len(preview)<max_sources:
                    preview.append(by_label[y][positions[y]]);positions[y]+=1
        selected=preview
    if synthetic_generator is None:
        if not torch.cuda.is_available():raise RuntimeError('Qwen generation requires allocated CUDA')
        if not Path(c['model_snapshot']).is_dir():raise FileNotFoundError('Local Qwen snapshot required; no download')
        from diffusers import QwenImageEditPlusPipeline
        pipe=QwenImageEditPlusPipeline.from_pretrained(c['model_snapshot'],torch_dtype=torch.bfloat16,local_files_only=True)
        pipe.enable_model_cpu_offload();pipe.set_progress_bar_config(disable=True)
        def render(source,condition,width,height,seed):
            with torch.inference_mode():
                return pipe(image=source,prompt=PROMPTS[condition],negative_prompt=' ',true_cfg_scale=c['generation_cfg'],
                    guidance_scale=1.,num_inference_steps=c['generation_steps'],width=width,height=height,
                    generator=torch.Generator(device='cpu').manual_seed(seed)).images[0].convert('RGB')
    else:render=synthetic_generator
    generated=0;seconds=0.
    for s in selected:
        folder=root/'items'/s['source_id'];folder.mkdir(parents=True,exist_ok=True)
        lock=folder/'WRITER.lock'
        try:lock.mkdir()
        except FileExistsError:raise RuntimeError('Source locked; verify writer status before explicit recovery: '+str(lock))
        try:
            if sha(s['original_image'])!=s['original_sha256']:raise ValueError('Original changed')
            record_path=folder/'record.json'
            record=json.loads(record_path.read_text()) if record_path.exists() else {'source':s,'plan_sha256':sha(root/'plan.json'),'variants':{}}
            if record['source']!=s or record['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Changed generation provenance')
            with Image.open(s['original_image']) as im:original=im.convert('RGB')
            w,h=original.size;ratio=w/h
            width=round(math.sqrt(c['generation_area']*ratio)/32)*32
            height=round(math.sqrt(c['generation_area']/ratio)/32)*32
            if min(width,height)<32:raise ValueError('Unsupported extreme aspect ratio')
            for condition in PROMPTS:
                dest=folder/(condition+'.png')
                if condition in record['variants']:
                    v=record['variants'][condition]
                    if v['image']!=str(dest) or sha(dest)!=v['sha256']:raise ValueError('Completed generation changed')
                    continue
                if dest.exists():raise ValueError('Unrecorded partial output preserved: '+str(dest))
                seed=candidate_seed(c['seed'],s['source_id']);start=time.perf_counter()
                native=render(original.copy(),condition,width,height,seed)
                if not isinstance(native,Image.Image):raise TypeError('Generator must return PIL image')
                native.convert('RGB').resize((w,h),Image.Resampling.BOX).save(dest)
                elapsed=time.perf_counter()-start;generated+=1;seconds+=elapsed
                record['variants'][condition]={'pair_id':s['source_id']+'_'+condition,'image':str(dest),
                    'sha256':sha(dest),'input_image':s['original_image'],'input_sha256':s['original_sha256'],
                    'independent_from_original':True,'condition':condition,'seed':seed,
                    'native_size':list(native.size),'requested_size':[width,height],'seconds':elapsed,'approved':False}
                atomic_json(record_path,record)
            resize=folder/'resize_geometry.png'
            if 'resize_control' not in record:
                if resize.exists():raise ValueError('Unrecorded resize control preserved')
                original.resize((width,height),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.BOX).save(resize)
                record['resize_control']={'image':str(resize),'sha256':sha(resize),'note':'Output-size geometry only; not a VAE/no-change reconstruction'}
                atomic_json(record_path,record)
            elif sha(resize)!=record['resize_control']['sha256']:raise ValueError('Resize control changed')
            print('GENERATED_OR_VERIFIED',s['source_id'],len(record['variants']),flush=True)
        finally:lock.rmdir()
    summary={'new_generated_outputs':generated,'new_generation_seconds':seconds,
        'seconds_per_new_output':seconds/generated if generated else None,'selected_sources':len(selected),
        'planned_total_outputs':plan['generated_outputs'],'approvals_created':0,'training_started':False}
    print(json.dumps(summary),flush=True);return summary


REVIEW_FIELDS=('pair_id','original_image','candidate_image','condition','approved','reviewer',
    'identity_preserved','attribute_valid','forensic_label_preserved','unrelated_details_preserved','evidence','comments')


def export_review(plan_dir,out,synthetic=False):
    root,plan,sources=load_plan(plan_dir,synthetic);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with out.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=REVIEW_FIELDS);writer.writeheader()
        for s in sources:
            p=root/'items'/s['source_id']/'record.json'
            if not p.exists():continue
            r=json.loads(p.read_text())
            for condition,v in r['variants'].items():
                writer.writerow({'pair_id':v['pair_id'],'original_image':s['original_image'],'candidate_image':v['image'],
                    'condition':condition,'approved':0,'identity_preserved':0,'attribute_valid':0,
                    'forensic_label_preserved':0,'unrelated_details_preserved':0})
                count+=1
    return {'pending_rows_exported':count,'review':str(out),'auto_approval':False}


def audit_bank(plan_dir,review_path,c,synthetic=False):
    rows,val,_,_=data_preflight(c,synthetic);root,plan,sources=load_plan(plan_dir,synthetic)
    if plan['config']!=c:raise ValueError('Generation plan/config mismatch; use original unbound configuration')
    if [s['original_image'] for s in sources]!=[r['image'] for r in rows]:raise ValueError('Plan must cover exact full TRAIN inventory')
    original_paths={r['image'] for r in rows+val};source_lookup={r['image']:r for r in rows}
    with Path(review_path).open(newline='') as f:
        reader=csv.DictReader(f)
        if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):raise ValueError('Missing review fields')
        reviews=list(reader)
    qa={}
    for q in reviews:
        if q['pair_id'] in qa:raise ValueError('Duplicate reviewed pair ID')
        qa[q['pair_id']]=q
    bank={};missing=Counter();used_ids=set();used_paths=set();approved_count=0
    for s in sources:
        source=s['original_image'];r=source_lookup[source]
        if s['label']!=r['label'] or any(s[k]!=r[k] for k in ('identity_ids','source_ids')):raise ValueError('Source provenance mismatch')
        if sha(source)!=s['original_sha256']:raise ValueError('Original bytes changed')
        path=root/'items'/s['source_id']/'record.json'
        if not path.exists():missing['ungenerated_source']+=1;continue
        record=json.loads(path.read_text())
        if record['source']!=s or record['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Record provenance mismatch')
        entry={'original_sha256':s['original_sha256'],'attributes':{a:[] for a in ATTRIBUTES},'nochange':None}
        for condition,v in record['variants'].items():
            if condition not in PROMPTS or v['condition']!=condition or v['pair_id']!=s['source_id']+'_'+condition:raise ValueError('Wrong candidate identity')
            if v['input_image']!=source or v['input_sha256']!=s['original_sha256'] or v['independent_from_original'] is not True:raise ValueError('Chained/wrong-source edit rejected')
            dest=str(Path(v['image']).resolve())
            expected=str((root/'items'/s['source_id']/(condition+'.png')).resolve())
            if dest!=expected or dest in original_paths or dest in used_paths:raise ValueError('Invalid or duplicate candidate path')
            used_paths.add(dest)
            if sha(dest)!=v['sha256']:raise ValueError('Candidate bytes changed')
            with Image.open(source) as im:source_size=im.size
            with Image.open(dest) as im:
                im.load()
                if im.size!=source_size:raise ValueError('Candidate geometry mismatch')
            q=qa.get(v['pair_id'])
            if q is None:continue
            used_ids.add(v['pair_id'])
            if q['original_image']!=source or q['candidate_image']!=dest or q['condition']!=condition:raise ValueError('Review association changed')
            if not flag(q['approved']):continue
            if not q['reviewer'].strip() or not q['evidence'].strip():raise ValueError('Approved edit needs reviewer and validity evidence')
            if not all(flag(q[k]) for k in ('identity_preserved','attribute_valid','forensic_label_preserved','unrelated_details_preserved')):
                raise ValueError('Approved edit failed explicit validity requirements')
            approved_count+=1
            candidate={'pair_id':v['pair_id'],'image':dest,'sha256':v['sha256'],'condition':condition}
            if condition=='nochange':entry['nochange']=candidate
            else:
                attr=next(a for a,ds in DIRECTIONS.items() if condition in ds)
                entry['attributes'][attr].append(candidate)
        for attr in ATTRIBUTES:
            entry['attributes'][attr].sort(key=lambda v:v['condition'])
            if not entry['attributes'][attr]:missing[attr]+=1
        if entry['nochange'] is None:missing['nochange']+=1
        rc=record.get('resize_control')
        if not rc or sha(rc['image'])!=rc['sha256']:raise ValueError('Missing/changed geometric control')
        entry['resize_control']=rc;bank[source]=entry
    if set(qa)-used_ids:raise ValueError('Review contains unknown/unassociated pair IDs')
    complete=not missing and len(bank)==len(rows)
    summary={'status':'PASS' if complete else 'BLOCKED_INCOMPLETE_REVIEWED_COVERAGE',
        'train_sources':len(rows),'sources_with_generation_records':len(bank),'approved_candidates':approved_count,
        'missing_source_counts':dict(missing),'required_attributes':list(ATTRIBUTES),
        'minimum_approved_variants_per_source_per_attribute':1,'nochange_required_per_source':True,
        'source_and_candidate_bytes_verified':True,'review_is_not_independent_scientific_proof':True}
    return bank,summary


def bind(c,plan_dir,review,pretrain_run,out,synthetic=False):
    bank,coverage=audit_bank(plan_dir,review,c,synthetic)
    if coverage['status']!='PASS':raise ValueError(json.dumps(coverage))
    pretrain_run=Path(pretrain_run).resolve();p=pretrain_run/'best.pt'
    if not (pretrain_run/'DONE').is_file() or not (pretrain_run/'RUN_VERIFIED.json').is_file():
        raise ValueError('Need completed independently verified pretraining')
    ck=torch.load(p,map_location='cpu',weights_only=True)
    summary=json.loads((pretrain_run/'summary.json').read_text())
    receipt=json.loads((pretrain_run/'RUN_VERIFIED.json').read_text())
    if ck['format']!=VERSION or ck['stage']!='pretrain' or ck['config']!=c or ck['source_sha256']!=sha(__file__):
        raise ValueError('Pretraining format/config/source mismatch')
    if bool(ck['synthetic_fixture'])!=synthetic:raise ValueError('Synthetic/production checkpoint mismatch')
    if summary['completed_epochs']!=c['pretrain_epochs'] or receipt['status']!='PASS' or receipt['checkpoint_sha256']!=sha(p):
        raise ValueError('Incomplete/changed pretraining')
    out=fresh(out)
    payload={'format':VERSION,'synthetic_fixture':synthetic,'plan_sha256':sha(Path(plan_dir)/'plan.json'),
        'review_sha256':sha(review),'train_sha256':c['input_hashes']['train'],'val_sha256':c['input_hashes']['val'],
        'coverage':coverage,'sources':bank}
    write(out/'pair_bank.json',payload)
    bound={**c,'pair_bank':str(out/'pair_bank.json'),'pair_bank_sha256':sha(out/'pair_bank.json'),
        'pretrain_checkpoint':str(p),'pretrain_checkpoint_sha256':sha(p)}
    write(out/'bound_config.json',bound);write(out/'coverage.json',coverage)
    return bound


def load_bank(c,rows,val,synthetic=False):
    if not c['pair_bank'] or not c['pretrain_checkpoint']:raise ValueError('Run bind after full coverage review and 50-epoch pretraining')
    payload=json.loads(Path(c['pair_bank']).read_text())
    if payload['format']!=VERSION or bool(payload['synthetic_fixture'])!=synthetic:raise ValueError('Invalid bank format/fixture role')
    if payload['train_sha256']!=c['input_hashes']['train'] or payload['val_sha256']!=c['input_hashes']['val']:
        raise ValueError('Bank split mismatch')
    bank=payload['sources'];forbidden={r['image'] for r in rows+val};used=set()
    if set(bank)!={r['image'] for r in rows}:raise ValueError('100% TRAIN source coverage is mandatory')
    for source,e in bank.items():
        if sha(source)!=e['original_sha256']:raise ValueError('Changed original')
        if set(e['attributes'])!=set(ATTRIBUTES) or not all(e['attributes'][a] for a in ATTRIBUTES) or not e['nochange']:
            raise ValueError('Missing approved per-source attribute/control')
        for v in [e['nochange'],e['resize_control']]+[v for a in ATTRIBUTES for v in e['attributes'][a]]:
            dest=str(Path(v['image']).resolve())
            if dest in forbidden or dest in used or sha(dest)!=v['sha256']:raise ValueError('Invalid/changed/duplicate bank variant')
            used.add(dest)
        for a in ATTRIBUTES:
            if any(v['condition'] not in DIRECTIONS[a] for v in e['attributes'][a]):raise ValueError('Wrong variant attribute')
    return bank


def pair_choice(source,entry,seed,epoch):
    offset=int(hashlib.sha256(f'{seed}:{source}'.encode()).hexdigest()[:12],16)
    attr=ATTRIBUTES[(epoch+offset)%len(ATTRIBUTES)]
    variants=entry['attributes'][attr]
    variant=variants[(epoch//len(ATTRIBUTES)+offset)%len(variants)]
    return attr,variant


class EpochBatches:
    """Shuffled full pass, without replacement; merge a singleton tail for BN."""
    def __init__(self,n,batch_size,seed,epoch):self.n,self.batch_size,self.seed,self.epoch=n,batch_size,seed,epoch
    def batches(self):
        indices=list(range(self.n));random.Random(f'{self.seed}:originals:{self.epoch}').shuffle(indices)
        batches=[indices[i:i+self.batch_size] for i in range(0,self.n,self.batch_size)]
        if len(batches)>1 and len(batches[-1])==1:
            tail=batches.pop();batches[-1].extend(tail)
        return batches
    def __iter__(self):return iter(self.batches())
    def __len__(self):return len(self.batches())


class OwnPairs(Dataset):
    def __init__(self,rows,weights,c,stage,epoch,bank=None):
        self.rows,self.weights,self.c,self.stage,self.epoch,self.bank=rows,weights,c,stage,epoch,bank
        self.reader=PairImages([],size=c['size'],backbone=c['backbone'])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];sample={'x':self.reader.load(r['image']),'y':float(r['label']),
            'weight':self.weights[i],'index':i,'attribute':'none','condition':'none','pair_id':''}
        if self.stage in ('cf','nochange'):
            entry=self.bank[r['image']];attr,v=pair_choice(r['image'],entry,self.c['seed'],self.epoch)
            actual=v if self.stage=='cf' else entry['nochange']
            sample.update(cf=self.reader.load(actual['image']),attribute=attr,condition=v['condition'],pair_id=actual['pair_id'])
        return sample


def objective(z,y,weights,zd=None,lambda_d=1.):
    if z.ndim!=1 or z.shape!=y.shape or z.shape!=weights.shape or (weights<=0).any():raise ValueError('Invalid paired loss input')
    classification=(weights*F.binary_cross_entropy_with_logits(z,y,reduction='none')).mean()
    invariance=z.sum()*0 if zd is None else (weights*(torch.asinh(z)-torch.asinh(zd)).abs()).mean()
    return {'total':classification+lambda_d*invariance,'classification':classification,'invariance':invariance}


def model_for(c,stage,device,synthetic=False):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    seed_all(c['seed']);torch.set_num_threads(4)
    model=DiagramCES(c['backbone'],c['weights'],c['hidden'],c['gate_hidden'],pretrained=stage=='pretrain',
        encoder_policy='trainable',bn_policy='update' if stage=='pretrain' else 'frozen',head_seed=c['head_seed'])
    if stage!='pretrain':
        ck=torch.load(c['pretrain_checkpoint'],map_location='cpu',weights_only=True)
        expected={**c,'pair_bank':None,'pair_bank_sha256':None,'pretrain_checkpoint':None,'pretrain_checkpoint_sha256':None}
        if ck['format']!=VERSION or ck['stage']!='pretrain' or ck['config']!=expected or ck['source_sha256']!=sha(__file__):
            raise ValueError('Pretraining transfer mismatch')
        if bool(ck['synthetic_fixture'])!=synthetic:raise ValueError('Synthetic/production checkpoint mismatch')
        model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in ck['model'].items() if k.startswith('encoder.')},strict=True)
    return model.to(device)


def load_model(path,device,synthetic=False):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['format']!=VERSION or bool(ck['synthetic_fixture'])!=synthetic:raise ValueError('Invalid checkpoint/fixture role')
    c=ck['config'];model=DiagramCES(c['backbone'],hidden=c['hidden'],gate_hidden=c['gate_hidden'],pretrained=False,
        encoder_policy='trainable',bn_policy='frozen',head_seed=c['head_seed'])
    model.load_state_dict(ck['model'],strict=True);return model.to(device).eval(),ck


def verify_run(out,device,synthetic=False):
    out=Path(out);model,ck=load_model(out/'best.pt',device,synthetic);c=ck['config']
    if sha(c['val'])!=c['input_hashes']['val']:raise ValueError('Validation changed')
    rows=read_rows(c['val']);z=predict(model,rows,device,c['batch_size'],c['workers'],c['size'])
    report=metrics(rows,z,ck['calibration']['threshold_logit'],c['min_support'])
    values=[]
    def compare(a,b):
        if isinstance(a,dict):
            if set(a)!=set(b):raise ValueError('Replay report structure changed')
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(float,int)) and not isinstance(a,bool):values.append(abs(a-b))
        elif a!=b:raise ValueError('Replay report value changed')
    compare(ck['validation'],report)
    ok=max(values)<1e-5
    receipt={'status':'PASS' if ok else 'FAIL','checkpoint_sha256':sha(out/'best.pt'),
        'source_sha256':sha(__file__),'max_metric_difference':max(values),'validation_replayed':True,
        'test_read':False,'synthetic_fixture':synthetic,'fairness_improvement_verified':False}
    write(out/('RUN_VERIFIED.json' if ok else 'VERIFICATION_FAILED.json'),receipt)
    if not ok:raise RuntimeError('Checkpoint replay mismatch')
    return receipt


def train(c,stage,out,device,resume=False,synthetic=False):
    if stage not in ('pretrain','bce','nochange','cf'):raise ValueError('Unknown stage')
    rows,val,weights,split=data_preflight(c,synthetic)
    if stage=='pretrain' and (c['pair_bank'] or c['pretrain_checkpoint']):raise ValueError('Use unbound pretraining config')
    bank=None if stage=='pretrain' else load_bank(c,rows,val,synthetic)
    model=model_for(c,stage,device,synthetic)
    opt=build_optimizer(model,{**c,'encoder_lr':c['pretrain_encoder_lr'] if stage=='pretrain' else c['encoder_lr']})
    epochs=c['pretrain_epochs'] if stage=='pretrain' else c['downstream_epochs']
    out=Path(out).resolve();start=0;best=-math.inf;best_epoch=None;pair_usage=Counter()
    if resume:
        if (out/'DONE').exists():raise ValueError('Completed runs immutable')
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if ck['source_sha256']!=sha(__file__) or ck['config']!=c or ck['stage']!=stage or ck['synthetic_fixture']!=synthetic:
            raise ValueError('Resume source/config/stage mismatch')
        history=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in history]!=list(range(1,ck['epoch']+1)):raise ValueError('Partial checkpoint/history; explicit recovery required')
        selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        if selected['epoch']!=ck['best_epoch'] or selected['config']!=c or selected['source_sha256']!=sha(__file__):
            raise ValueError('Selected checkpoint mismatch after interruption')
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        start,best,best_epoch=ck['epoch'],ck['best_auc'],ck['best_epoch'];pair_usage.update(ck['pair_usage'])
    else:
        fresh(out);write(out/'config.json',c)
        write(out/'provenance.json',{'format':VERSION,'source_sha256':sha(__file__),'stage':stage,'synthetic_fixture':synthetic,
            'parameters':model.counts(),'split_audit':split,'sampling':'shuffled full pass without replacement',
            'classification':'inverse label x group cell frequency weighting; full-dataset normalization',
            'encoder_policy':'all parameters trainable','bn_policy':'update' if stage=='pretrain' else 'frozen running buffers',
            'pairing':'own-source reviewed variant, rotating age/gender/skin; no separate pair pool',
            'pretraining_checkpoint':c['pretrain_checkpoint'],'input_hashes':c['input_hashes'],
            'test_read':False,'downstream_inputs_verified':bank is not None})
    started=time.perf_counter()
    for epoch in range(start,epochs):
        seed_all(c['seed']+epoch);model.train()
        dataset=OwnPairs(rows,weights,c,stage,epoch,bank)
        loader=DataLoader(dataset,batch_sampler=EpochBatches(len(rows),c['batch_size'],c['seed'],epoch),num_workers=c['workers'],
            generator=torch.Generator().manual_seed(c['seed']+epoch))
        seen=Counter();attr=Counter();directions=Counter();cell_exposure=Counter();pairs=0;forwards=0;sums=Counter()
        for step,b in enumerate(loader):
            n=len(b['y']);x=b['x'].to(device);paired=stage in ('cf','nochange')
            # Separate forwards keep original minibatch numerics matched to BCE; stage2 BN buffers are frozen.
            z=model(x)['logit'];zd=model(b['cf'].to(device))['logit'] if paired else None
            loss=objective(z,b['y'].float().to(device),b['weight'].float().to(device),zd,c['lambda_d'])
            if not torch.isfinite(loss['total']):raise RuntimeError('Nonfinite objective')
            opt.zero_grad(set_to_none=True);loss['total'].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise RuntimeError('Missing/nonfinite trainable gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            seen.update(b['index'].tolist());forwards+=n*(2 if paired else 1)
            for k,v in loss.items():sums[k]+=float(v.detach())*n
            if paired:
                pairs+=n;attr.update(b['attribute']);directions.update(b['condition']);pair_usage.update(b['pair_id'])
                for i,a in zip(b['index'].tolist(),b['attribute']):
                    row=rows[i];cell_exposure[f"{row['label']}:{row[c['balance_axis']]}:{a}"]+=1
            if (step+1)%100==0:print(json.dumps({'stage':stage,'epoch':epoch+1,'step':step+1,'steps':len(loader)}),flush=True)
        if set(seen)!=set(range(len(rows))) or any(v!=1 for v in seen.values()):raise RuntimeError('Original exposure violated')
        if stage in ('cf','nochange') and pairs!=len(rows):raise RuntimeError('Own-source pair coverage violated')
        z=predict(model,val,device,c['batch_size'],c['workers'],c['size']);y=np.asarray([r['label'] for r in val])
        calibration=calibrate(z[y==0],c['target_fpr']);report=metrics(val,z,calibration['threshold_logit'],c['min_support'])
        auc=report['overall']['auc'];improved=auc>best
        if improved:best,best_epoch=auc,epoch+1
        record={'epoch':epoch+1,'stage':stage,'steps':len(loader),'validation':report,'calibration':calibration,
            'losses':{k:v/len(rows) for k,v in sums.items()},'unique_originals':len(seen),'originals_once_each':True,
            'own_source_pair_count':pairs,'attributes':dict(attr),'scheduled_directions':dict(directions),
            'pair_attribute_by_label_group':dict(cell_exposure),
            'image_forwards':forwards,'selection':'validation_auc_only','selected_this_epoch':improved,
            'elapsed_seconds':time.perf_counter()-started}
        ck={'format':VERSION,'source_sha256':sha(__file__),'config':c,'stage':stage,'epoch':epoch+1,'synthetic_fixture':synthetic,
            'model':model.state_dict(),'optimizer':opt.state_dict(),'calibration':calibration,'validation':report,
            'best_auc':best,'best_epoch':best_epoch,'pair_usage':dict(pair_usage)}
        if improved:atomic_checkpoint(out/'best.pt',ck)
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        atomic_checkpoint(out/'last.pt',ck)
        print(json.dumps({'stage':stage,'epoch':epoch+1,'val_auc':auc,'pairs':pairs,'sources':len(seen)}),flush=True)
    write(out/'summary.json',{'status':'TRAINED_PENDING_REPLAY','stage':stage,'completed_epochs':epochs,
        'best_epoch':best_epoch,'best_validation_auc':best,'test_evaluated':False,'synthetic_fixture':synthetic,
        'unique_variants_used':len(pair_usage),'total_pair_draws':sum(pair_usage.values())})
    verify_run(out,device,synthetic)
    summary=json.loads((out/'summary.json').read_text());summary['status']='COMPLETED_VERIFIED';write(out/'summary.json',summary)
    (out/'DONE').write_text('Training and validation replay complete; no test evaluation.\n')
    return out/'best.pt'


def dry_run(c,stage,out,device,synthetic=False):
    rows,val,weights,_=data_preflight(c,synthetic);bank=None if stage=='pretrain' else load_bank(c,rows,val,synthetic)
    if stage not in ('pretrain','bce','nochange','cf'):raise ValueError('Unknown stage')
    out=fresh(out);model=model_for(c,stage,device,synthetic);model.train()
    b=next(iter(DataLoader(OwnPairs(rows,weights,c,stage,0,bank),batch_sampler=EpochBatches(len(rows),c['batch_size'],c['seed'],0),num_workers=0)))
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    z=model(b['x'].to(device))['logit'];zd=model(b['cf'].to(device))['logit'] if stage in ('cf','nochange') else None
    loss=objective(z,b['y'].float().to(device),b['weight'].float().to(device),zd,c['lambda_d']);loss['total'].backward()
    if not torch.isfinite(loss['total']) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Dry-run gradients failed')
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),before[k])]
    if stage=='pretrain':
        if any(not any(t in k for t in ('running_mean','running_var','num_batches_tracked')) for k in changed):raise RuntimeError('Unexpected dry-run update')
    elif changed:raise RuntimeError('Downstream dry run changed parameters or BN buffers')
    result={'status':'PASS','stage':stage,'source_sha256':sha(__file__),'device':device,'synthetic_fixture':synthetic,
        'originals':len(b['y']),'pairs':len(b['y']) if zd is not None else 0,'parameters':model.counts(),
        'optimizer_steps':0,'pretrain_bn_buffers_updated_in_discarded_model':stage=='pretrain',
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'losses':{k:float(v.detach()) for k,v in loss.items()}}
    write(out/'RESULT.json',result);return result


@torch.inference_mode()
def export_evidence(checkpoint,images,out,device,synthetic=False):
    model,ck=load_model(checkpoint,device,synthetic);c=ck['config'];out=fresh(out);records=[]
    reader=PairImages([],size=c['size'],backbone=c['backbone'])
    for i,path in enumerate(images):
        r=model(reader.load(path)[None].to(device));raw=r['local_map'][0].cpu().numpy()
        with Image.open(path) as im:im=im.convert('RGB')
        up=F.interpolate(r['local_map'][:,None],size=(im.height,im.width),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        scale=max(float(np.max(abs(up))),1e-12);heat=Image.fromarray(diverging(up,scale))
        Image.blend(im,heat,.45).save(out/f'{i:05d}_overlay.png')
        np.savez_compressed(out/f'{i:05d}_evidence.npz',local_map=raw,patch_gate=r['patch_gate'][0].cpu().numpy())
        z=float(r['logit'][0]);g=float(r['global_contribution'][0]);local=float(raw.mean());bias=float(r['fusion_bias'])
        if not math.isclose(z,g+local+bias,rel_tol=1e-5,abs_tol=1e-5):raise RuntimeError('E-box decomposition failed')
        records.append({'source_sha256':sha(path),'logit':z,'sigmoid_score':float(torch.sigmoid(r['logit'][0])),
            'prediction':int(z>=ck['calibration']['threshold_logit']),'threshold_logit':ck['calibration']['threshold_logit'],
            'global_contribution':g,'local_contribution':local,'fusion_bias':bias,'display_scale':scale,
            'note':'Sigmoid score is not a calibrated probability; red local supports fake, blue supports real. Not a forgery mask.'})
    write(out/'scores.json',records);return records


def evaluate(checkpoint,manifest,out,device,synthetic=False):
    model,ck=load_model(checkpoint,device,synthetic);c=ck['config']
    if sha(manifest) in (c['input_hashes']['train'],c['input_hashes']['val']):raise ValueError('TRAIN/VAL cannot be called test')
    rows=read_rows(manifest);z=predict(model,rows,device,c['batch_size'],c['workers'],c['size'])
    result={'checkpoint_sha256':sha(checkpoint),'manifest_sha256':sha(manifest),'source_sha256':sha(__file__),
        'threshold_source':'stored selected-checkpoint validation only','calibration':ck['calibration'],
        'calibrated':metrics(rows,z,ck['calibration']['threshold_logit'],c['min_support']),
        'fixed_probability_0_5':metrics(rows,z,0.,c['min_support']),
        'note':'External test membership/provenance requires review; no test threshold tuning.'}
    out=fresh(out);write(out/'metrics.json',result);return result


def smoke(device,out):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no GPU pass claimed')
    out=fresh(out);seed_all(17);torch.set_num_threads(4);checks=[]
    def passed(name):checks.append(name);print('PASS',name,flush=True)
    def reject(fn):
        try:fn()
        except (ValueError,FileExistsError):return
        raise AssertionError('Invalid case was accepted')
    for n,batch in [(5,2),(4,2),(3,16),(17,16)]:
        batches=list(EpochBatches(n,batch,17,0));flat=[i for b in batches for i in b]
        assert sorted(flat)==list(range(n)) and all(len(b)>=2 for b in batches)
        assert batches==list(EpochBatches(n,batch,17,0))
    z=torch.tensor([.2,-.8],device=device,requires_grad=True)
    zd=torch.tensor([-.1,-.4],device=device,requires_grad=True)
    y=torch.tensor([1.,0.],device=device);w=torch.tensor([.5,1.5],device=device)
    loss=objective(z,y,w,zd)
    ref=np.mean(np.array([.5,1.5])*(np.logaddexp(0,[.2,-.8])-np.array([1.,0.])*[.2,-.8]+abs(np.arcsinh([.2,-.8])-np.arcsinh([-.1,-.4]))))
    assert abs(float(loss['total'].detach())-ref)<1e-6
    loss['total'].backward();assert torch.isfinite(z.grad).all() and torch.count_nonzero(zd.grad)==2
    assert calibrate([.7]*4,.25)['achieved_fp']==0
    passed('full_epoch_no_replacement_singleton_tail_and_independent_paired_loss_reference')
    with tempfile.TemporaryDirectory(prefix='cesfl_cf5050_smoke_') as temp:
        root=Path(temp).resolve();rng=np.random.default_rng(17);paths={}
        for split in ('train','val','test'):
            records=[]
            for i in range(4):
                image=root/f'{split}{i}.png';Image.fromarray(rng.integers(0,256,(75,75,3),dtype=np.uint8)).save(image)
                records.append({'image':str(image),'label':i%2,'identity_ids':split+str(i),'source_ids':split+str(i),
                    'ids8':str(i//2),'gender_race':str(i//2),'family':'synthetic'})
            path=root/(split+'.csv');paths[split]=str(path)
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        initial=DiagramCES(pretrained=False,hidden=8,gate_hidden=8)
        weights=root/'synthetic_weights.pt';torch.save(initial.encoder.state_dict(),weights);del initial
        c={**config_default(),'train':paths['train'],'val':paths['val'],'weights':str(weights),'protocol_overlap_file':None,
            'pretrain_epochs':1,'downstream_epochs':2,'hidden':8,'gate_hidden':8,'size':75,'batch_size':2,'workers':0,
            'min_support':1,'generation_area':128**2}
        c['input_hashes']={k:sha(c[k]) for k in ('train','val','weights')}
        reject(lambda:validate_config(c))
        reject(lambda:data_preflight({**c,'input_hashes':{**c['input_hashes'],'train':'0'*64}},True))
        plan=plan_generation(c,root/'edits',True);rows=read_rows(paths['train'])
        originals=[np.asarray(Image.open(r['image']).convert('RGB')) for r in rows];calls=[]
        def stub(source,condition,width,height,seed):
            arr=np.asarray(source)
            matches=[i for i,a in enumerate(originals) if np.array_equal(arr,a)]
            assert len(matches)==1,'Edit input was not an original'
            calls.append((matches[0],condition,seed))
            # Synthetic pixels are software fixtures, not valid demographic edits.
            shift=list(PROMPTS).index(condition)+1
            return Image.fromarray(np.roll(arr,shift,axis=0)).resize((width,height))
        first=generate(plan,0,2,synthetic_generator=stub);second=generate(plan,1,2,synthetic_generator=stub)
        assert first['new_generated_outputs']+second['new_generated_outputs']==28
        assert generate(plan,0,2,synthetic_generator=stub)['new_generated_outputs']==0
        for i in range(4):
            seen=[r for r in calls if r[0]==i]
            assert len(seen)==7 and len({r[2] for r in seen})==1
        review=root/'review.csv';export_review(plan,review,True)
        bank,coverage=audit_bank(plan,review,c,True);assert coverage['status']!='PASS'
        passed('independent_original_generation_shards_resume_shared_source_seed_and_no_auto_approval')
        with review.open() as f:review_rows=list(csv.DictReader(f))
        for r in review_rows:
            r.update(approved=1,reviewer='SYNTHETIC_TEST_ONLY',identity_preserved=1,attribute_valid=1,
                forensic_label_preserved=1,unrelated_details_preserved=1,evidence='Synthetic fixture for software only; not real validity.')
        def save_review(records,path=review):
            with path.open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=REVIEW_FIELDS);wr.writeheader();wr.writerows(records)
        save_review(review_rows);bank,coverage=audit_bank(plan,review,c,True);assert coverage['status']=='PASS'
        incomplete=[dict(r) for r in review_rows]
        for r in incomplete:
            if r['original_image']==rows[0]['image'] and r['condition'].startswith('skin_'):r['approved']=0
        save_review(incomplete);assert audit_bank(plan,review,c,True)[1]['status']!='PASS'
        bad=[dict(r) for r in review_rows];bad[0]['original_image']=rows[1]['image'];save_review(bad)
        reject(lambda:audit_bank(plan,review,c,True));save_review(review_rows)
        record_path=plan/'items'/source_id(rows[0]['image'])/'record.json';original_record=json.loads(record_path.read_text())
        changed=json.loads(json.dumps(original_record));changed['variants']['older']['input_image']=changed['variants']['younger']['image']
        write(record_path,changed);reject(lambda:audit_bank(plan,review,c,True));write(record_path,original_record)
        bad=[dict(r) for r in review_rows];bad[0]['forensic_label_preserved']=0;save_review(bad)
        reject(lambda:audit_bank(plan,review,c,True));save_review(review_rows)
        passed('reject_missing_per_source_attributes_wrong_original_chained_edit_and_invalid_review')
        for r in rows:
            schedule=[pair_choice(r['image'],bank[r['image']],c['seed'],e) for e in range(6)]
            assert Counter(a for a,_ in schedule)=={a:2 for a in ATTRIBUTES}
            for a in ATTRIBUTES:assert {v['condition'] for attr,v in schedule if attr==a}==set(DIRECTIONS[a])
        dataset=OwnPairs(rows,[1.]*4,c,'cf',0,bank)
        for i,r in enumerate(rows):
            sample=dataset[i];assert sample['pair_id'].startswith(source_id(r['image'])+'_')
            assert torch.equal(sample['x'],dataset.reader.load(r['image']))
        passed('every_original_gets_its_own_edit_all_attributes_and_directions_rotate')
        dry_run(c,'pretrain',root/'dry_pretrain',device,True)
        train(c,'pretrain',root/'pretrain',device,synthetic=True)
        bound=bind(c,plan,review,root/'pretrain',root/'bound',True)
        reject(lambda:validate_config(bound))
        dry=dry_run(bound,'cf',root/'dry_cf',device,True);assert dry['pairs']==2 and dry['optimizer_steps']==0
        m1=model_for(bound,'bce',device,True);m2=model_for(bound,'cf',device,True)
        assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m1.state_dict().items())
        m1.train();assert all(not m.training for m in m1.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm))
        bn_before={k:v.clone() for k,v in m1.named_buffers()}
        result=m1(torch.randn(2,3,224,224,device=device))
        torch.testing.assert_close(result['logit'],result['global_contribution']+result['local_map'].mean((1,2))+result['fusion_bias'])
        assert all(torch.equal(v,bn_before[k]) for k,v in m1.named_buffers())
        assert m1.counts()['total']==m1.counts()['trainable']
        del m1,m2,result
        full_width=DiagramCES(pretrained=False,encoder_policy='trainable',bn_policy='frozen',head_seed=17).to(device)
        counts=full_width.counts();assert counts=={'total':21594159,'trainable':21594159}
        result=full_width(torch.randn(2,3,224,224,device=device))
        F.binary_cross_entropy_with_logits(result['logit'],torch.tensor([0.,1.],device=device)).backward()
        for branch in ('global_head','patch_head','image_gate','patch_gate'):
            grads=[p.grad for p in getattr(full_width,branch).parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads)>0,branch
        del full_width,result
        if device=='cuda':torch.cuda.empty_cache()
        passed('pretraining_transfer_identical_heads_full_trainability_frozen_downstream_bn_and_224px_ebox')
        train(bound,'cf',root/'cf',device,synthetic=True)
        original_save=globals()['atomic_checkpoint']
        def interrupted(path,value):
            original_save(path,value)
            if Path(path).name=='last.pt' and value['epoch']==1:raise InterruptedError('Synthetic epoch-boundary interruption')
        globals()['atomic_checkpoint']=interrupted
        try:
            try:train(bound,'cf',root/'resume',device,synthetic=True)
            except InterruptedError:pass
            else:raise AssertionError('Interruption did not occur')
        finally:globals()['atomic_checkpoint']=original_save
        train(bound,'cf',root/'resume',device,resume=True,synthetic=True)
        ck1=torch.load(root/'cf/last.pt',map_location='cpu',weights_only=True)
        ck2=torch.load(root/'resume/last.pt',map_location='cpu',weights_only=True)
        for k,v in ck1['model'].items():torch.testing.assert_close(v,ck2['model'][k],rtol=0,atol=0)
        assert ck1['pair_usage']==ck2['pair_usage'];del ck1,ck2
        for stage in ('bce','nochange'):train(bound,stage,root/stage,device,synthetic=True)
        for stage in ('bce','nochange','cf'):
            history=[json.loads(s) for s in (root/stage/'history.jsonl').read_text().splitlines()]
            assert all(h['unique_originals']==4 and h['originals_once_each'] for h in history)
            assert all(h['own_source_pair_count']==(0 if stage=='bce' else 4) for h in history)
        passed('all_three_training_arms_full_exposure_independent_replay_and_exact_resume')
        e1=evaluate(root/'cf/best.pt',paths['test'],root/'test1',device,True)
        with Path(paths['test']).open() as f:rr=list(csv.DictReader(f))
        for r in rr:r['label']=str(1-int(r['label']))
        flipped=root/'flipped.csv'
        with flipped.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
        e2=evaluate(root/'cf/best.pt',flipped,root/'test2',device,True)
        assert e1['calibration']==e2['calibration']
        reject(lambda:evaluate(root/'cf/best.pt',paths['val'],root/'wrong_test',device,True))
        export_evidence(root/'cf/best.pt',[rr[0]['image']],root/'maps',device,True)
        passed('frozen_test_threshold_label_flip_rejection_of_val_as_test_and_evidence_export')
    receipt={'status':'PASS','device':device,'checks':checks,'check_count':len(checks),'parameters':counts,
        'source_sha256':sha(__file__),'torch':str(torch.__version__),'python':sys.version.split()[0],
        'synthetic_fixture_only':True,'real_qwen_generation_tested':False,'real_training_tested':False,
        'cuda_tested':device=='cuda','fairness_improvement_verified':False}
    write(out/'RESULT.json',receipt);print(json.dumps(receipt,indent=2),flush=True);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init-config');p.add_argument('--out',required=True)
    p=sub.add_parser('plan');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('generate');p.add_argument('--plan',required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);p.add_argument('--max-sources',type=int)
    p=sub.add_parser('review-template');p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('audit-coverage');p.add_argument('--config',required=True);p.add_argument('--plan',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('bind');p.add_argument('--config',required=True);p.add_argument('--plan',required=True);p.add_argument('--review',required=True)
    p.add_argument('--pretrain-run',required=True);p.add_argument('--out',required=True)
    for command in ('train','dry-run'):
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['pretrain','bce','nochange','cf'],required=True)
        p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        if command=='train':p.add_argument('--resume',action='store_true')
    p=sub.add_parser('smoke');p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p=sub.add_parser('export');p.add_argument('--checkpoint',required=True);p.add_argument('--images',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=parser.parse_args();c=json.loads(Path(a.config).read_text()) if hasattr(a,'config') else None
    if a.command=='init-config':
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(json.dumps(config_default(),indent=2)+'\n')
    elif a.command=='plan':plan_generation(c,a.out)
    elif a.command=='generate':generate(a.plan,a.shard_index,a.num_shards,a.max_sources)
    elif a.command=='review-template':print(json.dumps(export_review(a.plan,a.out)))
    elif a.command=='audit-coverage':
        _,report=audit_bank(a.plan,a.review,c);write(fresh(a.out)/'coverage.json',report);print(json.dumps(report,indent=2))
        if report['status']!='PASS':sys.exit(2)
    elif a.command=='bind':print(json.dumps(bind(c,a.plan,a.review,a.pretrain_run,a.out),indent=2))
    elif a.command=='train':print(train(c,a.stage,a.out,a.device,a.resume))
    elif a.command=='dry-run':print(json.dumps(dry_run(c,a.stage,a.out,a.device),indent=2))
    elif a.command=='smoke':smoke(a.device,a.out)
    elif a.command=='evaluate':evaluate(a.checkpoint,a.manifest,a.out,a.device)
    else:export_evidence(a.checkpoint,a.images,a.out,a.device)


if __name__=='__main__':main()
