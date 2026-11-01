"""Read-only observer for authorized frozen-checkpoint test evaluation."""
import datetime, fcntl, importlib.util, json, os, time
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z')
spec=importlib.util.spec_from_file_location('publication_helpers',Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py'))
U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U);U.ROOT=ROOT
BASE='CES_FL/results/'+ROOT.name;STATUS='CES_FL/status/'+ROOT.name+'_live.md'
def observe():
    p=U.read(ROOT/'protocol.json');arms={}
    for arm in p['arms']:
        submission=U.read(ROOT/'runtime'/f'{arm}.submission.json');progress=U.read(ROOT/'runtime'/f'{arm}.progress.json')
        exit=U.read(ROOT/'runtime'/f'{arm}.exit.json');failure=U.read(ROOT/'runtime'/f'{arm}.failure.json')
        results={}
        for dataset in p['dataset_order']:
            d=ROOT/'evaluation'/arm/dataset
            metric=U.read(d/'metrics.json');receipt=U.read(d/'EVALUATION_VERIFIED.json')
            state='DONE' if (d/'DONE').exists() and receipt and receipt.get('status')=='PASS' else 'INFERENCE_DONE_CI_PENDING' if (d/'INFERENCE_DONE').exists() else 'PENDING'
            if state=='PENDING' and progress and progress.get('dataset')==dataset:state=progress['phase']
            if state=='PENDING' and failure:state='BLOCKED_BY_ARM_FAILURE'
            results[dataset]={'state':state,'metrics':metric,'confidence_intervals':U.read(d/'confidence_intervals.json'),'verification':receipt,'output':str(d)}
        replay=U.read(ROOT/'validation_audit'/arm/'replay.json')
        if replay:replay={k:v for k,v in replay.items() if k!='replayed_validation'}
        arms[arm]={'submission':submission,'progress':progress,'exit':exit,'failure':failure,
            'validation_replay':replay,'datasets':results,'log':str(ROOT/'logs'/f'{arm}.ssh.log')}
    return {'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':p,
        'software_checks':U.read(ROOT/'EVALUATION_SOFTWARE_PASS.json'),'arms':arms,'run_root':str(ROOT),
        'new_training':False,'test_thresholds_changed':False,'fairforge2_in_this_batch':'FF++-trained transfer evaluation',
        'ff2_indomain_plan':'/home/sachin.chaudhary/ces_fl_pipeline/training_plans/ff2_indomain_s17_20261101T042828Z'}
def render(d):
    p=d['protocol'];lines=['# CES-FL seed17 — authorized test evaluation','',
        'Observed UTC: **'+d['observed_utc']+'**. All five training runs completed; these are inference/uncertainty tasks.','',
        '**FairForge-2 below is FF++-trained transfer testing.** A separate FairForge-2 public in-domain training plan uses its own TRAIN/VAL/TEST.','',
        'Selected checkpoint hashes/epochs and validation thresholds were frozen before testing. No test-based calibration, model selection, gradient update or source/configuration change is performed.','',
        '| Arm | Dataset | State | Test AUC (%) | Actual test FPR (%) | ids8 FPR gap (pp) |',
        '|---|---|---|---:|---:|---:|']
    for arm,a in d['arms'].items():
        for dataset,r in a['datasets'].items():
            m=(r['metrics'] or {}).get('calibrated',{});o=m.get('overall',{});g=m.get('axes',{}).get('ids8',{})
            def val(v):return '—' if v is None else f'{100*v:.2f}'
            lines.append(f'| {arm} | {dataset} | {r["state"]} | {val(o.get("auc"))} | {val(o.get("fpr"))} | {val(g.get("fpr_gap"))} |')
    complete=sum(x['state']=='DONE' for a in d['arms'].values() for x in a['datasets'].values())
    lines+=['',f'Completed evaluations: **{complete}/25**. CI_PENDING means inference finished but uncertainty is still running.','',
        '[Complete aggregate evidence, all group/family metrics, CIs, hashes and process IDs](../../'+BASE+'/aggregate.json).','',
        'Inference uses the unchanged875bce model source. The external evaluation wrapper was tested on CPU and both CUDA nodes: three tests each, covering official-exporter agreement, immutable state/thresholds and exact native duplicate handling. Every selected checkpoint also replays validation before its test loop (AUC tolerance1e-5; calibration remains frozen).','',
        'Locked TEST row counts: FF++25,401; DFDC22,857; Celeb-DF28,458; DFD9,385; FairForge-287,564. Native DFDC includes1,476exact duplicate excess rows; their original weighting is preserved through the already audited unique index. Conflicting metadata is rejected.','',
        'Both fixed probability0.5 and stored validation-real empirical-budget thresholds are reported. Actual test FPR is measured; the requested validation target does not guarantee a test FPR. Undefined/unsupported subgroup metrics remain undefined.','',
        'CIs use1,000fixed-score bootstrap replicates of connected recorded-contributor components, seed17. Fewer than two components yields no independent-component CI. Hidden donors, proxy identities, sparse groups, single-seed training and multiple comparisons limit conclusions.','',
        'RN/RD used only six reviewed training sources. A fairness win or direct equivalence with Ding is not established merely by a test AUC or a single favorable group metric. The historical gate remains FAIL.','',
        'Execution follows the authorized SSH route on gpu01/gpu02, bounded shared VRAM,10%PyTorch allocator cap and no claimed PBS allocation. Existing jobs are untouched.','',
        'The finite24h observer reads/publishes only these already launched tasks. It cannot launch/retry/cancel training, generate images or change the calibration. Images, weights and row-level predictions remain private on cluster storage.','',
        'Cluster root: '+str(ROOT)]
    failures={a:v['failure'] for a,v in d['arms'].items() if v['failure']}
    if failures:lines+=['','Failures are recorded in aggregate evidence and the named private logs; no automatic retries.']
    return '\n'.join(lines)+'\n'
def main():
    lock=(ROOT/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    U.write(ROOT/'observer.json',{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'max_hours':24})
    started=time.monotonic();last=0;previous=None
    while time.monotonic()-started<24*3600:
        d=observe();U.write(ROOT/'CURRENT.json',d)
        signature=json.dumps({a:{'phase':v['progress'],'results':{k:x['state'] for k,x in v['datasets'].items()},'failure':v['failure']} for a,v in d['arms'].items()},sort_keys=True)
        if signature!=previous or time.monotonic()-last>1800:
            try:
                body=json.dumps(d,indent=2,allow_nan=False)+'\n';(ROOT/'AGGREGATE.json').write_text(body);(ROOT/'STATUS.md').write_text(render(d))
                c1=U.github_file(BASE+'/aggregate.json',body);c2=U.github_file(STATUS,render(d))
                U.write(ROOT/'last_publication.json',{'aggregate_commit':c1,'status_commit':c2,'utc':d['observed_utc'],'status_path':STATUS})
                print('GITHUB_PROGRESS_PUSH '+c1+' '+c2,flush=True);last=time.monotonic();previous=signature
            except Exception as e:print('PUBLICATION_ERROR '+repr(e),flush=True)
        if all(v['failure'] or all(x['state']=='DONE' for x in v['datasets'].values()) for v in d['arms'].values()):return
        time.sleep(30)
if __name__=='__main__':main()
