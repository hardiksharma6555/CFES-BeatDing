"""Controlled CES-FL upgrade. No training is triggered by import or --help.

Reuses the immutable v0.2.1 architecture and data/metric utilities. Adds partial
encoder training, stochastic group DRO, and independently sampled age/control
pairs. These are exploratory options, not verified fairness improvements.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import ces_fl_diagram_model_v021 as base

VERSION = 'CESFL_UPGRADE_v0.3_experimental'
BASE_SHA = '0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de'


class UpgradeCES(base.DiagramCES):
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


class GroupDRO(nn.Module):
    """Exponentiated group weights; absent minibatch groups have zero loss.

    Group means use originals only. Update q with detached losses, then use q to
    weight differentiable group means. Persist q in checkpoints. This is the
    stochastic group-DRO surrogate, not a direct FPR-parity constraint.
    """
    def __init__(self, n_groups, eta):
        super().__init__()
        if n_groups < 1 or not math.isfinite(eta) or eta <= 0:
            raise ValueError('Invalid group DRO settings')
        self.eta = eta
        self.register_buffer('log_q', torch.full((n_groups,), -math.log(n_groups)))

    def forward(self, losses, group_ids):
        if losses.ndim != 1 or losses.shape != group_ids.shape or not len(losses):
            raise ValueError('Invalid group loss alignment')
        if group_ids.dtype != torch.long or group_ids.min() < 0 or group_ids.max() >= len(self.log_q):
            raise ValueError('Invalid group ids')
        if not torch.isfinite(losses).all():
            raise ValueError('Nonfinite group losses')
        counts = torch.bincount(group_ids, minlength=len(self.log_q))
        means = losses.new_zeros(len(self.log_q)).index_add(0, group_ids, losses) / counts.clamp_min(1)
        with torch.no_grad():
            self.log_q.add_(self.eta * means.detach())
            self.log_q.sub_(torch.logsumexp(self.log_q, dim=0))
        return (self.log_q.exp().clone() * means).sum()


class PairPool:
    """Fixed-size, class-balanced pair draws with independent RNG and cycling.

    Sample original sources uniformly within label × observed-group cells.
    Rotate through each source's approved age directions. No-change and age arms
    share anchors and direction schedules; only the selected variant differs.
    Repeated anchors are permitted and explicitly logged (small-pool pilot).
    """
    def __init__(self, rows, pairs, controls, axis, seed, count, minimum_sources):
        if count < 2 or count % 2:
            raise ValueError('Pair count must be positive and even')
        self.count, self.seed, self.draws = count, seed, Counter()
        self.controls = {r['original_image']: r for r in controls}
        self.rows = {r['image']: r for r in rows}
        self.cells = defaultdict(list)
        self.pairs = {}
        for source, groups in sorted(pairs.items()):
            edits = sorted(groups.get('age', []), key=lambda p: (p.get('direction', ''), p['pair_id']))
            if not edits:
                continue
            if {p.get('direction') for p in edits} != {'older', 'younger'}:
                raise ValueError('Every eligible source needs reviewed older AND younger directions')
            self.pairs[source] = edits
            row = self.rows[source]
            self.cells[(row['label'], row[axis])].append(source)
        if len(self.pairs) < minimum_sources:
            raise ValueError(f'Need {minimum_sources} approved sources; found {len(self.pairs)}')
        self.keys = {label: sorted(k for k in self.cells if k[0] == label) for label in (0, 1)}
        if not all(self.keys.values()):
            raise ValueError('Pair pool must contain real AND fake originals')

    def select(self, epoch, step):
        rng = random.Random(f'{self.seed}:pairs:{epoch}:{step}')
        chosen = []
        for label in (0, 1):
            for _ in range(self.count // 2):
                cell = rng.choice(self.keys[label])
                source = rng.choice(self.cells[cell])
                edits = self.pairs[source]
                edit = edits[self.draws[source] % len(edits)]
                self.draws[source] += 1
                chosen.append((source, edit))
        rng.shuffle(chosen)
        return chosen

    def load(self, dataset, epoch, step, mode):
        if mode not in ('age', 'nochange'):
            raise ValueError('Unknown consistency mode')
        chosen = self.select(epoch, step)
        x = torch.stack([dataset.load(source) for source, _ in chosen])
        xd = torch.stack([dataset.load(edit['counterfactual_image'] if mode == 'age'
                         else self.controls[source]['nochange_control']) for source, edit in chosen])
        return x, xd, chosen


def validate_config(c):
    required = {'seed','backbone','weights','hidden','gate_hidden','size','epochs','batch_size','workers',
        'sampler','balance_axis','lr','encoder_lr','weight_decay','target_fpr','min_support',
        'train','val','protocol_overlap_file','encoder_checkpoint','encoder_checkpoint_sha256',
        'head_seed','encoder_policy','classification','dro_eta','dro_min_group_count',
        'consistency','pair_count','minimum_pair_sources','lambda_d','pairs','controls_manifest',
        'input_hashes','purpose'}
    if set(c) != required:
        raise ValueError(f'Configuration keys differ: missing={required-set(c)}, extra={set(c)-required}')
    if c['backbone'] not in ('xception','resnet50') or c['encoder_policy'] not in ('frozen','partial'):
        raise ValueError('Unsupported architecture policy')
    if c['classification'] not in ('bce','group_dro') or c['consistency'] not in ('none','age','nochange'):
        raise ValueError('Unsupported objective')
    if c['sampler'] != 'balanced' or c['balance_axis'] not in base.AXES:
        raise ValueError('Controlled arms require the same balanced original sampler')
    for k in ('epochs','batch_size','hidden','gate_hidden','min_support','dro_min_group_count','minimum_pair_sources'):
        if type(c[k]) is not int or c[k] < 1: raise ValueError('Invalid '+k)
    if type(c['size']) is not int or c['size'] < 75 or type(c['workers']) is not int or c['workers'] < 0:
        raise ValueError('Invalid image size/workers')
    for k in ('seed','head_seed','pair_count'):
        if type(c[k]) is not int or c[k] < 0: raise ValueError('Invalid '+k)
    for k in ('lr','encoder_lr','dro_eta'):
        if not math.isfinite(c[k]) or c[k] <= 0: raise ValueError('Invalid '+k)
    for k in ('weight_decay','lambda_d','target_fpr'):
        if not math.isfinite(c[k]) or c[k] < 0: raise ValueError('Invalid '+k)
    if c['target_fpr'] >= 1: raise ValueError('Invalid calibration target')
    if c['consistency'] == 'none':
        if c['pair_count'] or c['lambda_d'] or c['pairs'] or c['controls_manifest']:
            raise ValueError('No-pair arms must not silently load/train pairs')
    elif c['pair_count'] < 2 or c['pair_count'] % 2 or not c['pairs'] or not c['controls_manifest'] or c['lambda_d'] <= 0:
        raise ValueError('Pair arms require even positive pair count, controls and positive lambda')


def preflight(c):
    validate_config(c)
    if base.sha(base.__file__) != BASE_SHA:
        raise ValueError('Historical utility source changed; re-review before use')
    keys = {'train','val','weights','encoder_checkpoint'}
    keys |= {k for k in ('protocol_overlap_file','pairs','controls_manifest') if c[k]}
    if set(c['input_hashes']) != keys:
        raise ValueError('Every consumed external input must have an exact SHA256')
    for k in keys:
        if base.sha(c[k]) != c['input_hashes'][k]: raise ValueError('Changed input: '+k)
    if c['encoder_checkpoint_sha256'] != c['input_hashes']['encoder_checkpoint']:
        raise ValueError('Encoder hash disagreement')
    rows, val = base.read_rows(c['train']), base.read_rows(c['val'])
    splits = base.check_splits(rows, val, c['protocol_overlap_file'])
    groups = sorted({(r['label'],r[c['balance_axis']]) for r in rows})
    index = {g:i for i,g in enumerate(groups)}
    ids = [index[(r['label'],r[c['balance_axis']])] for r in rows]
    counts = Counter(ids)
    if c['classification'] == 'group_dro' and min(counts.values()) < c['dro_min_group_count']:
        raise ValueError('Insufficient original TRAIN support for a DRO group; do not silently drop it')
    pool = None
    if c['consistency'] != 'none':
        pairs, _ = base.read_pairs(c['pairs'], rows, ['age'])
        controls = base.read_controls(c['controls_manifest'], rows, pairs)
        forbidden = {r['image'] for r in rows+val}
        variants = [p['counterfactual_image'] for g in pairs.values() for p in g.get('age',[])]
        variants += [r[k] for r in controls for k in ('nochange_control','resize_control','condition_resize_control')]
        if any(str(Path(p).resolve()) in forbidden for p in variants):
            raise ValueError('An edit/control overlaps an original TRAIN/validation path')
        pool = PairPool(rows, pairs, controls, c['balance_axis'], c['seed'],
                        c['pair_count'], c['minimum_pair_sources'])
    return rows, val, ids, groups, splits, pool


def run(c, out, device, *, _synthetic_smoke=False):
    rows,val,group_ids,groups,splits,pool = preflight(c)
    out = base.empty_out(out)
    base.seed_all(c['seed']); torch.set_num_threads(4)
    model = UpgradeCES(c['backbone'],c['weights'],c['hidden'],c['gate_hidden'],
                       pretrained=not _synthetic_smoke,encoder_policy=c['encoder_policy'],head_seed=c['head_seed']).to(device)
    imported = None if _synthetic_smoke else base.import_encoder(model,{**c,'stage':'downstream'})
    optimizer = base.build_optimizer(model,c)
    dro = GroupDRO(len(groups),c['dro_eta']).to(device)
    counts = Counter(group_ids)
    source_hashes = {'upgrade':base.sha(__file__),'base':base.sha(base.__file__)}
    provenance = {'format':VERSION,'source_sha256':source_hashes,'input_hashes':c['input_hashes'],
        'splits':splits,'group_order':groups,'group_counts':[counts[i] for i in range(len(groups))],
        'parameters':model.counts(),'selected_encoder_modules':list(model.selected_names),
        'bn_policy':'all running buffers frozen; selected affine parameters may train',
        'encoder_import':imported,'synthetic_smoke':_synthetic_smoke,'test_read':False,
        'pair_sources':len(pool.pairs) if pool else 0,
        'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]}
    base.write(out/'config.json',c); base.write(out/'provenance.json',provenance)
    best = -math.inf; usage = Counter(); start=time.perf_counter()
    for epoch in range(c['epochs']):
        model.train()
        dataset = base.PairImages(rows,size=c['size'],backbone=c['backbone'])
        sampler = WeightedRandomSampler([1/counts[g] for g in group_ids],len(rows),replacement=True,
                    generator=torch.Generator().manual_seed(c['seed']+epoch))
        loader = DataLoader(dataset,batch_size=c['batch_size'],sampler=sampler,num_workers=c['workers'],
                            collate_fn=base.collate,generator=torch.Generator().manual_seed(c['seed']+epoch))
        classification_sum=0.; original_count=0; inv_sum=0.; pairs_count=0; image_forwards=0
        pair_labels=Counter(); pair_groups=Counter(); directions=Counter()
        for step,b in enumerate(loader):
            x=b['x']; n=len(x); parts=[x]; chosen=[]
            if pool:
                anchors,edits,chosen=pool.load(dataset,epoch,step,c['consistency'])
                parts += [anchors,edits]
            output=model(torch.cat(parts).to(device)); image_forwards+=sum(len(t) for t in parts)
            y=b['y'].to(device); z=output['logit']
            individual=F.binary_cross_entropy_with_logits(z[:n],y,reduction='none')
            gids=torch.tensor([group_ids[i] for i in b['rows']],device=device)
            classification = dro(individual,gids) if c['classification']=='group_dro' else individual.mean()
            k=len(chosen)
            inv=(torch.asinh(z[n:n+k])-torch.asinh(z[n+k:n+2*k])).abs().mean() if k else z.sum()*0
            loss=classification+c['lambda_d']*inv
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradients')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            optimizer.step()
            classification_sum+=float(classification.detach())*n; original_count+=n
            inv_sum+=float(inv.detach())*k; pairs_count+=k
            for source,edit in chosen:
                usage[edit['pair_id']]+=1
                row=pool.rows[source]; pair_labels[str(row['label'])]+=1
                pair_groups[f"{row['label']}:{row[c['balance_axis']]}"]+=1
                directions[edit['direction']]+=1
        scores=base.predict(model,val,device,c['batch_size'],c['workers'],c['size'])
        labels=np.asarray([r['label'] for r in val])
        calibration=base.calibrate(scores[labels==0],c['target_fpr'])
        report=base.metrics(val,scores,calibration['threshold_logit'],c['min_support'])
        record={'epoch':epoch+1,'steps':step+1,'validation':report,'calibration':calibration,
            'selection':'validation_auc_only','classification':classification_sum/original_count,
            'invariance':inv_sum/pairs_count if pairs_count else None,'pair_count':pairs_count,
            'pair_labels':dict(pair_labels),'pair_groups':dict(pair_groups),'pair_directions':dict(directions),
            'image_forwards':image_forwards,'dro_weights':dro.log_q.exp().detach().cpu().tolist(),
            'elapsed_seconds':time.perf_counter()-start}
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        checkpoint={'format':VERSION,'model':model.state_dict(),'config':c,'epoch':epoch+1,
            'optimizer':optimizer.state_dict(),'dro':dro.state_dict(),'group_order':groups,
            'calibration':calibration,'validation':report,'source_sha256':source_hashes,
            'input_hashes':c['input_hashes'],'synthetic_smoke':_synthetic_smoke}
        if report['overall']['auc']>best:
            best=report['overall']['auc']; torch.save(checkpoint,out/'best.pt')
        torch.save(checkpoint,out/'last.pt')
        base.write(out/'pair_usage.json',{'scheduled_age_pair_draws':dict(usage),
                   'source_draws':dict(pool.draws) if pool else {},'consistency_mode':c['consistency']})
        print(json.dumps({'epoch':epoch+1,'validation_auc':report['overall']['auc'],'pairs':pairs_count}),flush=True)
    (out/'DONE').write_text('Training finished; no test evaluated; independent result review required\n')
    return out/'best.pt'


def load_model(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('format')!=VERSION: raise ValueError('Not an upgrade checkpoint')
    c=ck['config']; model=UpgradeCES(c['backbone'],hidden=c['hidden'],gate_hidden=c['gate_hidden'],
                                     pretrained=False,encoder_policy=c['encoder_policy'],head_seed=c['head_seed'])
    model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck


def evaluate(checkpoint,manifest,out,device):
    model,ck=load_model(checkpoint,device)
    rows=base.read_rows(manifest); c=ck['config']
    scores=base.predict(model,rows,device,c['batch_size'],c['workers'],c['size'])
    out=base.empty_out(out)
    report={'checkpoint_sha256':base.sha(checkpoint),'manifest_sha256':base.sha(manifest),
        'threshold_source':'selected checkpoint validation only','calibration':ck['calibration'],
        'synthetic_smoke':ck['synthetic_smoke'],
        'calibrated':base.metrics(rows,scores,ck['calibration']['threshold_logit'],c['min_support']),
        'fixed_probability_0_5':base.metrics(rows,scores,0.,c['min_support'])}
    base.write(out/'metrics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','train-config'):
        p=sub.add_parser(name);p.add_argument('--config',required=True)
        if name=='train-config': p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.command=='evaluate': evaluate(args.checkpoint,args.manifest,args.out,args.device)
    else:
        c=json.loads(Path(args.config).read_text())
        if args.command=='train-config':run(c,args.out,args.device)
        else:
            rows,val,ids,groups,splits,pool=preflight(c)
            print(json.dumps({'status':'PASS','train':len(rows),'val':len(val),'groups':groups,
                'splits':splits,'pair_sources':len(pool.pairs) if pool else 0,'training_started':False}))


if __name__=='__main__':main()
