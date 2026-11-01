"""Launch exactly one named stage over SSH; stage receipts prevent duplicates."""
import json, subprocess, shlex, datetime
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z')
ARMS=['R0_frozen_bce','R1_partial_bce','R2_partial_dro','RN_partial_dro_nochange_pilot','RD_partial_dro_age_pilot']
def launch(stage,host):
    assert stage in ['SMOKE_gpu01','SMOKE_gpu02']+ARMS and host in ['gpu01','gpu02']
    receipt=ROOT/'runtime'/f'{stage}.submission.json'
    if receipt.exists():return json.loads(receipt.read_text())
    attempt=ROOT/'runtime'/f'{stage}.attempt.json'
    assert not attempt.exists(),'Prior uncertain launch; do not retry automatically'
    log=ROOT/'logs'/f'{stage}.ssh.log'
    assert not log.exists()
    worker=str(ROOT/'runtime/gpu_eval_worker.py')
    code="import subprocess,json,sys,pathlib,os; f=pathlib.Path("+repr(str(log))+").open('xb'); e=dict(os.environ); e['PATH']='/home/sachin.chaudhary/ces_fl_tools/git_cluster_20261031/bin:/opt/pbs/bin:/usr/bin:/bin:'+e.get('PATH',''); e.pop('PBS_JOBID',None); child=subprocess.Popen([sys.executable,'-u',"+repr(worker)+",'--stage',"+repr(stage)+"],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=e); print(json.dumps({'worker_pid':child.pid,'host':"+repr(host)+",'stage':"+repr(stage)+"}))"
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'/home/sachin.chaudhary/.conda/envs/aiface/bin/python -c '+shlex.quote(code)]
    attempt.write_text(json.dumps({'command':command,'stage':stage,'host':host,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    p=subprocess.run(command,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    record={**json.loads(p.stdout),'command':command,'log':str(log),'execution_mode':'direct_ssh_bounded_shared_gpu','pbs_allocation':None,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    return record
if __name__=='__main__':
    for host in ['gpu01','gpu02']:
        print(json.dumps(launch('SMOKE_'+host,host)),flush=True)
