"""Frozen-checkpoint inference, validation replay and locked native test evaluation."""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def modules(protocol):
    source=Path(protocol['source'])
    for rel,digest in protocol['source_sha256'].items():assert sha(source/rel)==digest,'Changed evaluation source: '+rel
    sys.path.insert(0,str(source/'CES_FL/src'))
    import ces_fl_upgrade as upgrade
    path=source/'CES_FL/experiments/benchmark_seed17/uncertainty.py'
    spec=importlib.util.spec_from_file_location('locked_component_bootstrap',path);unc=importlib.util.module_from_spec(spec);spec.loader.exec_module(unc)
    return upgrade,unc
def rows_for(M,dataset,path,audit):
    assert sha(path)==audit['sha256'],'Changed test manifest'
    if not audit['native_exact_duplicate_extra_rows']:
        rows=M.read_rows(path)
    else:
        assert dataset=='dfdc','Only the previously audited exact DFDC repetitions are supported'
        unique=audit['unique_validation_index'];assert sha(unique)==audit['unique_validation_index_sha256']
        parsed=M.read_rows(unique);index={r['image']:r for r in parsed}
        with Path(unique).open(newline='') as f:original={str(Path(r['image']).resolve()):r for r in csv.DictReader(f)}
        with Path(path).open(newline='') as f:native=list(csv.DictReader(f))
        rows=[]
        for row in native:
            key=str(Path(row['image']).resolve())
            assert row==original[key],'Conflicting duplicate/native metadata'
            rows.append(dict(index[key]))
        assert len(rows)-len(index)==audit['native_exact_duplicate_extra_rows']
    assert len(rows)==audit['rows']
    return rows
def state_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def reports(M,rows,scores,calibration,minimum):
    before=json.dumps(calibration,sort_keys=True)
    report={'threshold_source':'selected checkpoint validation only','calibration':dict(calibration),
        'calibrated':M.metrics(rows,scores,calibration['threshold_logit'],minimum),
        'fixed_probability_0_5':M.metrics(rows,scores,0.,minimum)}
    assert json.dumps(calibration,sort_keys=True)==before,'Metrics changed calibration'
    return report
def evaluate_dataset(up,unc,protocol,arm,model,ck,dataset,rows,out,*,reps=None):
    assert not Path(out).exists(),'Preserve existing evaluation outputs'
    out=Path(out);out.mkdir(parents=True)
    original=state_hash(model);threshold=dict(ck['calibration']);c=ck['config']
    started=time.perf_counter()
    scores=up.base.predict(model,rows,next(model.parameters()).device,c['batch_size'],c['workers'],c['size'])
    assert len(scores)==len(rows) and np.isfinite(scores).all(),'Nonfinite/misaligned predictions'
    assert state_hash(model)==original,'Inference changed model parameters or buffers'
    assert all(p.grad is None for p in model.parameters()),'Unexpected evaluation gradients'
    assert ck['calibration']==threshold==protocol['arms'][arm]['calibration'],'Test changed threshold'
    report=reports(up.base,rows,scores,threshold,c['min_support'])
    report.update(checkpoint_sha256=protocol['arms'][arm]['sha256'],
        manifest_sha256=protocol['manifest_audit'][dataset]['sha256'],synthetic_smoke=bool(protocol.get('synthetic_smoke',False)),
        native_exact_duplicate_extra_rows_preserved=protocol['manifest_audit'][dataset]['native_exact_duplicate_extra_rows'],
        inference_elapsed_seconds=time.perf_counter()-started,score_space='raw binary fake-class logit')
    write(out/'metrics.json',report)
    np.savez_compressed(out/'predictions_private.npz',logit=scores,label=np.asarray([r['label'] for r in rows],dtype=np.int8))
    with (out/'predictions_private.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['image','label','logit','prediction'])
        w.writerows((r['image'],r['label'],float(z),int(z>=threshold['threshold_logit'])) for r,z in zip(rows,scores))
    (out/'INFERENCE_DONE').write_text('Inference completed; threshold unchanged; uncertainty pending\n')
    started=time.perf_counter()
    ci=unc.bootstrap(rows,scores,threshold['threshold_logit'],up.base.AXES,
        reps=protocol['bootstrap']['replicates'] if reps is None else reps,
        seed=protocol['bootstrap']['seed'],min_support=c['min_support'])
    write(out/'confidence_intervals.json',ci)
    receipt={'status':'PASS','arm':arm,'dataset':dataset,'rows':len(rows),'checkpoint_sha256':report['checkpoint_sha256'],
        'manifest_sha256':report['manifest_sha256'],'model_state_unchanged':True,'no_gradients':True,
        'threshold_unchanged':True,'test_calibration_or_selection':False,'inference_only':True,
        'bootstrap_elapsed_seconds':time.perf_counter()-started,'threshold_source':report['threshold_source']}
    write(out/'EVALUATION_VERIFIED.json',receipt);(out/'DONE').write_text('Fixed selected checkpoint/threshold evaluated; no training or test-based calibration\n')
    return report
def run(protocol_path,arm,device='cuda'):
    protocol=json.loads(Path(protocol_path).read_text());root=Path(protocol['run_root']);assert arm in protocol['arms']
    up,unc=modules(protocol);up.base.seed_all(protocol['seed']);torch.set_num_threads(4)
    entry=protocol['arms'][arm];assert sha(entry['checkpoint'])==entry['sha256']
    model,ck=up.load_model(entry['checkpoint'],device)
    assert ck['calibration']==entry['calibration'] and ck['epoch']==entry['selected_epoch'] and ck['synthetic_smoke'] is False
    assert sha(entry['config_path'])==entry['config_sha256']
    assert ck['config']==json.loads(Path(entry['config_path']).read_text())
    progress=lambda phase,**extra:write(root/'runtime'/f'{arm}.progress.json',{'phase':phase,'arm':arm,'host':os.uname().nodename,'time':time.time(),**extra})
    progress('VALIDATION_REPLAY')
    assert sha(ck['config']['val'])==ck['input_hashes']['val']
    validation=up.base.read_rows(ck['config']['val'])
    before=state_hash(model)
    scores=up.base.predict(model,validation,device,ck['config']['batch_size'],ck['config']['workers'],ck['config']['size'])
    validation_report=reports(up.base,validation,scores,ck['calibration'],ck['config']['min_support'])
    delta=abs(validation_report['calibrated']['overall']['auc']-ck['validation']['overall']['auc'])
    assert delta<1e-5,'Selected checkpoint validation-AUC replay mismatch'
    assert state_hash(model)==before
    write(root/'validation_audit'/arm/'replay.json',{'status':'PASS','recorded_validation_auc':ck['validation']['overall']['auc'],
        'replayed_validation':validation_report,'absolute_auc_difference':delta,'recalibration_performed':False,
        'checkpoint_sha256':entry['sha256'],'threshold_unchanged':True})
    for dataset in protocol['dataset_order']:
        progress('READING_LOCKED_TEST_MANIFEST',dataset=dataset)
        rows=rows_for(up.base,dataset,protocol['manifests'][dataset],protocol['manifest_audit'][dataset])
        destination=root/'evaluation'/arm/dataset
        progress('INFERENCE_AND_UNCERTAINTY',dataset=dataset,rows=len(rows))
        report=evaluate_dataset(up,unc,protocol,arm,model,ck,dataset,rows,destination)
        assert sha(entry['checkpoint'])==entry['sha256'],'Evaluation modified checkpoint file'
        print('TEST_RESULT',arm,dataset,json.dumps(report['calibrated']['overall']),flush=True)
    progress('DONE');(root/'runtime'/f'{arm}.EVALUATION_DONE').write_text('All five fixed-manifest evaluations and uncertainty completed\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--arm',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(a.protocol,a.arm,a.device)
