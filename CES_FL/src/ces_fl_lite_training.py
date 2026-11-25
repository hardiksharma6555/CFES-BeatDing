#!/usr/bin/env python3
"""Train and validate a compact CES-FL student using a frozen native teacher.

Commands: prepare, dry-run, train, export-int8, compare. No command reads TEST or starts
a scheduler job. Native data splits, interventions, error losses and metrics
are imported from the hash-pinned original implementation. The native trainer
is not modified. A CPU QNNPACK export is NOT an ESP-DL model.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ces_fl_head_ablation as native
import ces_fl_lite_feasibility as lite

VERSION = 'CESFL_LITE_TRAIN_v1'
NATIVE_SHA = '08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177'
ARMS = ('bce', 'kd', 'fair_response')


def source_hashes():
    paths = [Path(__file__), Path(native.__file__), Path(lite.__file__)]
    result = {p.name: native.sha(p) for p in paths}
    if result[Path(native.__file__).name] != NATIVE_SHA:
        raise ValueError('Native scientific source changed; review a new protocol')
    return result


class LiteDetector(nn.Module):
    """Native [-1,1] RGB crop -> ImageNet normalization -> compact fake logit.

    Preprocessing is inside the host artifact. Firmware must reproduce this
    exact preprocessing before replacing it with target-specific operations.
    All downstream BN running buffers are frozen; affine weights train.
    """
    def __init__(self, name):
        super().__init__()
        self.net = lite.make_model(name)
        self.size = lite.CANDIDATES[name]['size']
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        for m in self.net.modules():
            if isinstance(m, nn.Dropout): m.p = 0.
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        return self

    def forward(self, x):
        x = (x + 1.) / 2.
        if x.shape[-2] != self.size or x.shape[-1] != self.size:
            x = F.interpolate(x, size=[self.size, self.size], mode='bilinear', align_corners=False)
        z = self.net((x - self.mean) / self.std).squeeze(1)
        return {'logit': z}

    def counts(self):
        return {'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)}


def new_student(c, fixture=False):
    native.seed_all(c['seed'])
    model = LiteDetector(c['student'])
    if c['student_init']:
        sd = torch.load(c['student_init'], map_location='cpu', weights_only=True)
        # Explicit torchvision ImageNet trunk; binary head remains freshly seeded.
        trunk = {k: v for k, v in sd.items() if k.startswith('features.')}
        result = model.net.load_state_dict(trunk, strict=False)
        expected = {k for k in model.net.state_dict() if k.startswith('classifier.')}
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise ValueError('Student initialization is not a compatible ImageNet trunk')
    elif not fixture:
        raise ValueError('Production requires the bound local student initialization')
    return model


def check_config(c, fixture=False):
    required = {'format', 'dataset_id', 'student', 'student_init', 'student_init_sha256',
                'teacher_run', 'teacher_sha256', 'teacher_cache_sha256', 'teacher_preflight',
                'qualification', 'train', 'val', 'train_sha256', 'val_sha256',
                'overlap_file', 'overlap_sha256', 'seed', 'epochs', 'batch_size', 'workers',
                'lr', 'weight_decay', 'lambda_kd', 'lambda_response_kd', 'kd_temperature',
                'arm', 'max_train_seconds', 'quant_calibration_size', 'synthetic_fixture'}
    if set(c) != required:
        raise ValueError(f'Config mismatch: missing={required-set(c)}, extra={set(c)-required}')
    if c['format'] != VERSION or c['synthetic_fixture'] is not fixture:
        raise ValueError('Format or fixture scope differs')
    if c['arm'] not in ARMS or c['student'] not in lite.CANDIDATES:
        raise ValueError('Unknown arm/student')
    if not fixture and c['student'] != 'mobile_v2_relu_100':
        raise ValueError('First production comparison is preregistered at full-width 224')
    for k in ('epochs', 'batch_size', 'quant_calibration_size'):
        if type(c[k]) is not int or c[k] < 1: raise ValueError(k)
    for k in ('seed', 'workers'):
        if type(c[k]) is not int or c[k] < 0: raise ValueError(k)
    for k in ('lr', 'weight_decay', 'lambda_kd', 'lambda_response_kd', 'kd_temperature', 'max_train_seconds'):
        if not isinstance(c[k], (float, int)) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(k)
    if bool(c['student_init']) != bool(c['student_init_sha256']): raise ValueError('Incomplete initialization binding')
    if not fixture and not c['student_init']: raise ValueError('Local ImageNet initialization required')
    for k, hk in [('train','train_sha256'),('val','val_sha256'),('student_init','student_init_sha256'),('overlap_file','overlap_sha256')]:
        if c[k] and native.sha(c[k]) != c[hk]: raise ValueError(f'Changed {k}')
    source_hashes()
    return c


def teacher_evidence(c, fixture=False):
    root = Path(c['teacher_run'])
    required = ('best.pt','RUN_VERIFIED.json','DONE','config.json','history.json','validation_scores_PRIVATE.json')
    if any(not (root / n).is_file() for n in required): raise ValueError('Teacher is not a complete verified run')
    if native.sha(root/'best.pt') != c['teacher_sha256']: raise ValueError('Teacher bytes changed')
    if native.sha(root/'validation_scores_PRIVATE.json') != c['teacher_cache_sha256']: raise ValueError('Teacher cache changed')
    receipt = json.loads((root/'RUN_VERIFIED.json').read_text())
    tc = json.loads((root/'config.json').read_text())
    history = json.loads((root/'history.json').read_text())
    if receipt['status'] != 'COMPLETED_VERIFIED' or bool(receipt['synthetic_fixture']) != fixture:
        raise ValueError('Teacher completion/fixture mismatch')
    if receipt['source_sha256'] != NATIVE_SHA or receipt['config_digest'] != native.digest(tc):
        raise ValueError('Teacher provenance differs')
    if receipt['selected_checkpoint_sha256'] != c['teacher_sha256']:
        raise ValueError('Teacher receipt is bound to different bytes')
    if len(history) != tc['epochs'] or receipt['epochs_completed'] != tc['epochs']:
        raise ValueError('Teacher epoch budget incomplete')
    if [v['epoch'] for v in history] != list(range(1,tc['epochs']+1)):
        raise ValueError('Teacher history is non-contiguous')
    best = max(history, key=lambda r:r['validation']['overall']['auc'])
    if best['epoch'] != receipt['selected_epoch']: raise ValueError('Teacher selection rule changed')
    if best['validation'] != receipt['validation']: raise ValueError('Teacher selected metrics differ')
    for k in ('dataset_id','train_sha256','val_sha256','overlap_sha256'):
        if tc[k] != c[k]: raise ValueError(f'Teacher/student {k} differs')
    if tc['source_sha256'] != NATIVE_SHA: raise ValueError('Teacher config source differs')
    if c['arm'] == 'fair_response' and not fixture:
        native.check_qualification(tc, c['qualification'])
    rows, val, pre = native.checked_inputs(tc, c['teacher_preflight'])
    audit = native.split_audit(rows, val, c['overlap_file'])
    if native.sha(tc['train']) != c['train_sha256'] or native.sha(tc['val']) != c['val_sha256']:
        raise ValueError('Teacher input manifest bytes changed')
    # The private cache is the same native selected model and exact VAL cohort.
    cache = json.loads((root/'validation_scores_PRIVATE.json').read_text())
    if cache['split'] != 'validation' or cache['rows'] != val:
        raise ValueError('Teacher cache row order/metadata changed')
    if cache['checkpoint_sha256'] != c['teacher_sha256'] or cache['manifest_sha256'] != c['val_sha256']:
        raise ValueError('Teacher cache checkpoint/membership mismatch')
    replay = native.metric_report(val, cache['logits'], cache['threshold'], tc['min_eval_support'])
    if replay != receipt['validation']: raise ValueError('Teacher cache metrics do not replay')
    # New loader paths must exactly identify the same audited images and metadata.
    for given, inherited in [(c['train'],tc['train']),(c['val'],tc['val'])]:
        if Path(given).resolve() != Path(inherited).resolve():
            raise ValueError('Relocated manifests require an explicit provenance adapter')
    return tc, rows, val, cache, {'teacher_epoch':best['epoch'],'split_audit':audit,
                                'native_preflight_inventory_sha256':pre['inventory_sha256']}


def prepare(c, out, fixture=False):
    check_config(c, fixture)
    tc, rows, val, cache, provenance = teacher_evidence(c, fixture)
    model = new_student(c, fixture)
    out = native.fresh_dir(out)
    native.write_json(out/'config.json', c)
    record = {'status':'PREPARED', 'config_digest':native.digest(c),
              'source_hashes':source_hashes(), 'parameters':model.counts(),
              'train_rows':len(rows), 'validation_rows':len(val),
              'images_rehashed_during_load':True, 'full_fresh_image_decode_executed':False,
              'test_read':False, 'synthetic_fixture':fixture, **provenance}
    native.write_json(out/'PREPARED.json',record)
    return record


def prepared_inputs(directory, fixture=False):
    p=Path(directory); c=check_config(json.loads((p/'config.json').read_text()),fixture)
    record=json.loads((p/'PREPARED.json').read_text())
    if record['config_digest']!=native.digest(c) or record['source_hashes']!=source_hashes():
        raise ValueError('Preparation/source changed')
    tc,rows,val,cache,provenance=teacher_evidence(c,fixture)
    base={**tc,'seed':c['seed'],'epochs':c['epochs'],'batch_size':c['batch_size'],'workers':c['workers']}
    return c,base,rows,val,cache


def dry_run(prepared, out, device='cpu', fixture=False):
    """One discarded TRAIN backward pass; no optimizer update or VAL scoring."""
    c,base,rows,_,_=prepared_inputs(prepared,fixture)
    device=torch.device(device);torch.set_num_threads(4)
    student=new_student(c,fixture).to(device).train()
    teacher,_=native.load_trained(Path(c['teacher_run'])/'best.pt',device,fixture)
    teacher.eval().requires_grad_(False)
    # Bound workload: two real and two fake originals, plus the native joint aux.
    chosen=[]
    for label in (0,1):chosen.extend([i for i,r in enumerate(rows) if r['label']==label][:2])
    ds=native.Originals(rows,base,epoch=0,pairs=c['arm']=='fair_response')
    b=torch.utils.data.default_collate([ds[i] for i in chosen])
    active,_=native.group_inventory(rows,base)
    event=base['axes'].index('gender_race') if 'gender_race' in base['axes'] else 0
    aux=native.auxiliary(rows,active,base,event,0) if c['arm']=='fair_response' else None
    weights=native.original_weights(rows,base['balance_axis'])
    names=sorted({(r['label'],r[base['balance_axis']]) for r in rows});codes={n:i for i,n in enumerate(names)}
    cells=torch.tensor([codes[r['label'],r[base['balance_axis']]] for r in rows])
    # A TRAIN-only detached threshold from the two real originals suffices to
    # exercise this backward check; production uses the full fixed TRAIN subset.
    with torch.no_grad():
        logits=student(b['x'].to(device))['logit'].float().cpu().numpy()
    threshold=native.calibrate(logits[b['y'].numpy()==0],base['target_fpr'])['threshold_logit']
    before={k:v.clone() for k,v in student.named_buffers() if 'running_' in k or 'num_batches_tracked' in k}
    start=time.perf_counter()
    losses,pairs=distillation_step(student,teacher,b,aux,weights,base,c,cells,threshold,device,0)
    if not all(torch.isfinite(v) for v in losses.values()):raise RuntimeError('Nonfinite dry-run loss')
    losses['total'].backward()
    grads=[p.grad for p in student.parameters() if p.requires_grad]
    if not all(g is not None and torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid student gradients')
    if not sum(float(g.abs().sum()) for g in grads)>0:raise RuntimeError('Zero student update direction')
    if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('Teacher received gradients')
    buffers=dict(student.named_buffers())
    if any(not torch.equal(v,buffers[k]) for k,v in before.items()):raise RuntimeError('BN running state changed')
    result={'status':'PASS','source_hashes':source_hashes(),'config_digest':native.digest(c),
            'device':str(device),'main_train_images':len(chosen),'real_source_pairs':pairs,
            'native_auxiliary_axis':None if aux is None else aux['axis'],
            'losses':{k:float(v.detach()) for k,v in losses.items()},'seconds':time.perf_counter()-start,
            'student_gradients_finite_nonzero':True,'teacher_frozen':True,'student_bn_buffers_frozen':True,
            'optimizer_updates':0,'validation_inference':False,'test_read':False,'synthetic_fixture':fixture}
    out=native.fresh_dir(out);native.write_json(out/'DRY_RUN.json',result);return result


def distillation_step(student, teacher, batch, aux, weights, base, c, cells, threshold, device, epoch):
    # Reuse the unchanged H objective; the second forward below is explicitly
    # counted extra KD compute. Student dropout is zero and BN buffers are frozen.
    native_arm='D' if c['arm']=='fair_response' else 'A'
    losses,pairs=native.train_step(student,batch,aux,weights,base,native_arm,threshold,device,epoch)
    zero=losses['total']*0; kd=zero; rd=zero
    if c['arm']!='bce':
        x=batch['x'].to(device)
        with native.autocast(device,base):
            sz=student(x)['logit']
        with torch.no_grad():tz=teacher(x)['logit']
        codes=cells[batch['index']].to(device)
        if c['arm']=='kd': codes=torch.zeros_like(codes)  # ordinary sample-mean KL control
        kd=lite.logit_distillation(sz,tz,codes,c['kd_temperature'])
        if c['arm']=='fair_response':
            real=batch['y']==0
            if real.any():
                xs,xt=batch['s'][real].to(device),batch['t'][real].to(device)
                with native.autocast(device,base):
                    ss,st=student(xs)['logit'],student(xt)['logit']
                with torch.no_grad():ts,tt=teacher(xs)['logit'],teacher(xt)['logit']
                rd=lite.response_distillation(ss,st,ts,tt,codes[real.to(device)])
    losses={**losses,'teacher_logit_kd':kd,'teacher_response_kd':rd,
            'total':losses['total']+c['lambda_kd']*kd+c['lambda_response_kd']*rd}
    return losses,pairs


def load_student(path, device='cpu', fixture=False):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['format']!=VERSION or ck['source_hashes']!=source_hashes() or ck['synthetic_fixture'] is not fixture:
        raise ValueError('Student checkpoint format/source/fixture mismatch')
    model=LiteDetector(ck['config']['student']);model.load_state_dict(ck['model'],strict=True)
    return model.to(device).eval(),ck


def train(prepared, out, device, resume=False, fixture=False, stop_after=None):
    c,base,rows,val,teacher_cache=prepared_inputs(prepared,fixture)
    device=torch.device(device);torch.set_num_threads(4);native.seed_all(c['seed'])
    teacher,tck=native.load_trained(Path(c['teacher_run'])/'best.pt',device,fixture)
    teacher.requires_grad_(False);teacher.eval()
    if tck['epoch']!=json.loads((Path(c['teacher_run'])/'RUN_VERIFIED.json').read_text())['selected_epoch']:
        raise ValueError('Teacher checkpoint epoch differs')
    if tck['config']!=json.loads((Path(c['teacher_run'])/'config.json').read_text()):
        raise ValueError('Teacher checkpoint config differs')
    if c['arm']=='fair_response' and not fixture and tck['qualification_sha256']!=native.sha(c['qualification']):
        raise ValueError('Teacher-qualified intervention provenance changed')
    student=new_student(c,fixture).to(device)
    optimizer=torch.optim.AdamW(student.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    active,_=native.group_inventory(rows,base)
    weights=native.original_weights(rows,base['balance_axis'])
    cell_names=sorted({(r['label'],r[base['balance_axis']]) for r in rows})
    code={name:i for i,name in enumerate(cell_names)}
    cells=torch.tensor([code[r['label'],r[base['balance_axis']]] for r in rows])
    train_real=sorted([r for r in rows if r['label']==0],key=lambda r:native.seed_for(c['seed'],'train-calibration',r['image']))[:base['train_calibration_size']]
    out=Path(out);history=[];elapsed=0.;best=-math.inf;start=0
    if resume:
        if (out/'DONE').exists():raise ValueError('Completed run preserved')
        if (out/'BUDGET_STOP.json').exists():raise ValueError('Mid-epoch budget stop requires an explicit recovery plan')
        _,ck=load_student(out/'last.pt',device,fixture)
        if ck['config']!=c:raise ValueError('Resume config mismatch')
        if native.sha(out/'best.pt')!=ck['best_checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        student.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        history=ck['history'];elapsed=ck['elapsed_seconds'];start=ck['epoch'];best=ck['best_auc']
    else:
        native.fresh_dir(out);native.write_json(out/'config.json',c)
        native.write_json(out/'RUN.json',{'source_hashes':source_hashes(),'parameters':student.counts(),
                                        'teacher_sha256':c['teacher_sha256'],'test_read':False,'synthetic_fixture':fixture})
    started=time.perf_counter();previous=elapsed
    for epoch in range(start,c['epochs']):
        elapsed=previous+time.perf_counter()-started
        estimate=history[-1]['epoch_seconds'] if history else 0
        if elapsed+estimate>c['max_train_seconds'] or (stop_after is not None and epoch>=stop_after):
            result={'status':'INCOMPLETE_EPOCH_BOUNDARY_STOP','epochs_completed':epoch}
            native.write_json(out/'STATUS.json',result);return result
        tic=time.perf_counter();native.seed_all(c['seed']+epoch)
        cal=native.calibrate(native.predict(student,train_real,base,device),base['target_fpr'])
        student.train();sums=Counter();seen=Counter();pairs=0;steps=0
        event=epoch*math.ceil(math.ceil(len(rows)/c['batch_size'])/base['aux_every'])
        for step,b in enumerate(native.make_loader(rows,base,epoch,pairs=c['arm']=='fair_response',shuffle=True)):
            elapsed=previous+time.perf_counter()-started
            if elapsed>=c['max_train_seconds']:
                native.atomic_save(out/'partial_budget_stop.pt',{'format':VERSION,'config':c,
                    'source_hashes':source_hashes(),'model':student.state_dict(),'optimizer':optimizer.state_dict(),
                    'completed_history':history,'partial_epoch':epoch+1,'partial_steps':steps,
                    'elapsed_seconds':elapsed,'synthetic_fixture':fixture})
                native.write_json(out/'BUDGET_STOP.json',{'status':'STOPPED_MID_EPOCH','epoch':epoch+1,
                    'steps':steps,'elapsed_seconds':elapsed,'automatic_resume_allowed':False})
                raise RuntimeError('Compute budget exhausted; partial state preserved, dependent launches must stop')
            aux=None
            if step%base['aux_every']==0:
                if c['arm']=='fair_response':aux=native.auxiliary(rows,active,base,event,epoch)
                event+=1
            losses,n=distillation_step(student,teacher,b,aux,weights,base,c,cells,cal['threshold_logit'],device,epoch)
            native.finite_step(losses,student,optimizer)
            if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('Teacher received gradients')
            seen.update(b['index'].tolist());pairs+=n;steps+=1
            for k,v in losses.items():sums[k]+=float(v.detach())
            if steps%100==0:print(json.dumps({'epoch':epoch+1,'step':steps,'loss':float(losses['total'].detach())}),flush=True)
        if len(seen)!=len(rows) or any(n!=1 for n in seen.values()):raise RuntimeError('Original coverage changed')
        if c['arm']=='fair_response' and pairs!=sum(r['label']==0 for r in rows):raise RuntimeError('Real pair coverage differs')
        z=native.predict(student,val,base,device)
        calibration=native.calibrate(z[np.array([r['label']==0 for r in val])],base['target_fpr'])
        report=native.metric_report(val,z,calibration['threshold_logit'],base['min_eval_support'])
        improved=report['overall']['auc']>best
        if improved:best=report['overall']['auc']
        record={'epoch':epoch+1,'steps':steps,'originals_once_each':True,'unique_originals':len(seen),
                'real_source_pairs':pairs,'validation':report,'calibration':calibration,
                'train_calibration':cal,'selected_this_epoch':improved,
                'losses_step_mean':{k:v/steps for k,v in sums.items()},'epoch_seconds':time.perf_counter()-tic}
        history.append(record);elapsed=previous+time.perf_counter()-started
        ck={'format':VERSION,'source_hashes':source_hashes(),'config':c,'native_config':base,
            'model':student.state_dict(),'optimizer':optimizer.state_dict(),'history':history,
            'epoch':epoch+1,'best_auc':best,'validation':report,'calibration':calibration,
            'elapsed_seconds':elapsed,'synthetic_fixture':fixture}
        if improved:native.atomic_save(out/'best.pt',ck)
        ck['best_checkpoint_sha256']=native.sha(out/'best.pt');native.atomic_save(out/'last.pt',ck)
        native.write_json(out/'history.json',history)
        print(json.dumps({'arm':c['arm'],'epoch':epoch+1,'validation_auc':report['overall']['auc'],'seconds':elapsed}),flush=True)
    selected,sck=load_student(out/'best.pt',device,fixture)
    z=native.predict(selected,val,base,device)
    report=native.metric_report(val,z,sck['calibration']['threshold_logit'],base['min_eval_support'])
    if report!=sck['validation']:raise RuntimeError('Selected checkpoint replay differs')
    native.write_json(out/'validation_scores_PRIVATE.json',{'rows':val,'logits':z.tolist(),
        'threshold':sck['calibration']['threshold_logit'],'manifest_sha256':c['val_sha256'],
        'checkpoint_sha256':native.sha(out/'best.pt'),'min_support':base['min_eval_support'],
        'split':'validation','test_read':False})
    # Frozen FP32 inference artifact, without optimizer or private membership.
    selected=selected.cpu().eval();script=torch.jit.freeze(torch.jit.script(selected))
    torch.jit.save(script,str(out/'student_fp32.pt'))
    result={'status':'COMPLETED_VERIFIED','epochs_completed':c['epochs'],'selected_epoch':sck['epoch'],
            'validation':report,'fixed_probability_0_5':native.metric_report(val,z,0.,base['min_eval_support']),
            'teacher_validation':native.metric_report(val,teacher_cache['logits'],teacher_cache['threshold'],base['min_eval_support']),
            'selected_checkpoint_sha256':native.sha(out/'best.pt'),'fp32_file_bytes':(out/'student_fp32.pt').stat().st_size,
            'source_hashes':source_hashes(),'test_evaluated':False,'synthetic_fixture':fixture}
    native.write_json(out/'RUN_VERIFIED.json',result)
    (out/'DONE').write_text('Complete with independent selected VAL replay; TEST not read.\n')
    return result


def calibration_indices(rows, size, seed):
    cells={}
    for i,r in enumerate(rows):cells.setdefault((r['label'],r['ids8']),[]).append(i)
    if size<len(cells):raise ValueError('Calibration budget cannot cover every class/group cell')
    for key in cells:cells[key].sort(key=lambda i:native.seed_for(seed,'quant-calibration',rows[i]['image']))
    ordered=[];offset=0
    while len(ordered)<min(size,len(rows)):
        for key in sorted(cells):
            if offset<len(cells[key]):ordered.append(cells[key][offset])
            if len(ordered)==min(size,len(rows)):break
        offset+=1
    return ordered


def export_int8(prepared, run, out, backend='qnnpack', fixture=False):
    c,base,rows,val,teacher_cache=prepared_inputs(prepared,fixture)
    run=Path(run);receipt=json.loads((run/'RUN_VERIFIED.json').read_text())
    if not (run/'DONE').is_file() or receipt['status']!='COMPLETED_VERIFIED':raise ValueError('Student training not complete')
    if native.sha(run/'best.pt')!=receipt['selected_checkpoint_sha256']:raise ValueError('Student bytes changed')
    model,ck=load_student(run/'best.pt','cpu',fixture)
    if ck['config']!=c:raise ValueError('Export config differs')
    if backend not in torch.backends.quantized.supported_engines:raise ValueError('Quant backend unavailable')
    torch.set_num_threads(4);torch.backends.quantized.engine=backend
    model.net.fuse_model();model.net.qconfig=torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(model.net,inplace=True)
    selected=calibration_indices(rows,c['quant_calibration_size'],c['seed'])
    with torch.inference_mode():
        for i in selected:
            # Load asserts the old TRAIN inventory hash for each calibration image.
            model(native.normalize(native.load_pixels(rows[i],base['size']))[None])
    unused=lite.audit_observers(model.net)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');torch.ao.quantization.convert(model.net,inplace=True)
    if not any(isinstance(m,torch.ao.nn.quantized.Conv2d) for m in model.net.modules()):
        raise RuntimeError('No quantized convolutions')
    out=native.fresh_dir(out);script=torch.jit.freeze(torch.jit.script(model.eval()))
    torch.jit.save(script,str(out/'student_int8.pt'))
    loaded=torch.jit.load(str(out/'student_int8.pt')).eval()
    sample=native.normalize(native.load_pixels(rows[selected[0]],base['size']))[None]
    with torch.inference_mode():torch.testing.assert_close(loaded(sample)['logit'],model(sample)['logit'])
    # Evaluate the actual reloaded exported model, not a float surrogate.
    z=native.predict(loaded,val,base,torch.device('cpu'))
    calibration=native.calibrate(z[np.array([r['label']==0 for r in val])],base['target_fpr'])
    report=native.metric_report(val,z,calibration['threshold_logit'],base['min_eval_support'])
    record={'status':'EXPORTED_AND_VALIDATED','source_hashes':source_hashes(),
            'model_sha256':native.sha(out/'student_int8.pt'),'model_bytes':(out/'student_int8.pt').stat().st_size,
            'under_10_decimal_MB':(out/'student_int8.pt').stat().st_size<10000000,
            'quantization_backend':backend,'esp32_compatible_artifact':False,
            'calibration_split':'TRAIN','calibration_count':len(selected),
            'calibration_inventory_digest':native.digest([(rows[i]['image'],rows[i]['image_sha256']) for i in selected]),
            'unused_nonresidual_observers':unused,'conversion_warnings':sorted({str(w.message) for w in caught}),
            'validation':report,'calibration':calibration,'threshold_source':'INT8 model validation real only',
            'fixed_probability_0_5':native.metric_report(val,z,0.,base['min_eval_support']),
            'validation_at_float_threshold':native.metric_report(val,z,ck['calibration']['threshold_logit'],base['min_eval_support']),
            'test_read':False,'synthetic_fixture':fixture}
    native.write_json(out/'EXPORT_REPORT.json',record)
    native.write_json(out/'validation_scores_PRIVATE.json',{'rows':val,'logits':z.tolist(),'threshold':calibration['threshold_logit'],
        'manifest_sha256':c['val_sha256'],'checkpoint_sha256':native.sha(out/'student_int8.pt'),
        'min_support':base['min_eval_support'],'split':'validation','test_read':False})
    return record


def compare(first, second, out, replicates=1000):
    a,b=json.loads(Path(first).read_text()),json.loads(Path(second).read_text())
    if a.get('split')!='validation' or b.get('split')!='validation':raise ValueError('This study helper accepts VAL only')
    native.compare(first,second,out,replicates,17)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('prepare');s.add_argument('--config',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('train');s.add_argument('--prepared',required=True);s.add_argument('--out',required=True)
    s.add_argument('--device',default='cuda');s.add_argument('--resume',action='store_true')
    s=sub.add_parser('dry-run');s.add_argument('--prepared',required=True);s.add_argument('--out',required=True)
    s.add_argument('--device',default='cpu')
    s=sub.add_parser('export-int8');s.add_argument('--prepared',required=True);s.add_argument('--run',required=True)
    s.add_argument('--out',required=True);s.add_argument('--backend',choices=['qnnpack','fbgemm'],default='qnnpack')
    s=sub.add_parser('compare');s.add_argument('--first',required=True);s.add_argument('--second',required=True)
    s.add_argument('--out',required=True);s.add_argument('--replicates',type=int,default=1000)
    a=p.parse_args()
    if a.cmd=='prepare':result=prepare(json.loads(Path(a.config).read_text()),a.out)
    elif a.cmd=='dry-run':result=dry_run(a.prepared,a.out,a.device)
    elif a.cmd=='train':result=train(a.prepared,a.out,a.device,a.resume)
    elif a.cmd=='export-int8':result=export_int8(a.prepared,a.run,a.out,a.backend)
    else:result=compare(a.first,a.second,a.out,a.replicates)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
