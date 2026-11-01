"""Direct SSH execution with bounded spare VRAM; no PBS allocation is claimed."""
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z')
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672')
GIT='/home/sachin.chaudhary/ces_fl_tools/git_cluster_20261031/bin/git'
LOCKS=Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')
ARMS=['R0_frozen_bce','R1_partial_bce','R2_partial_dro','RN_partial_dro_nochange_pilot','RD_partial_dro_age_pilot']
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def call(args):return subprocess.check_output(args,text=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['SMOKE_gpu01','SMOKE_gpu02']+ARMS);stage=p.parse_args().stage
    host=socket.gethostname().split('.')[0];assert host in ['gpu01','gpu02']
    guard=(ROOT/'runtime'/f'{stage}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'runtime'/f'{stage}.exit.json').exists(),'Do not rerun an executed stage'
    assert call([GIT,'-C',str(SOURCE),'rev-parse','HEAD']).strip()=='875bce025084871438ef799ca9086985c4375672'
    assert not call([GIT,'-C',str(SOURCE),'status','--porcelain','--','CES_FL']).strip()
    expected=json.loads((ROOT/'handoff_verification.json').read_text())['sha256']
    for path,digest in expected.items():
        assert hashlib.sha256((SOURCE/path).read_bytes()).hexdigest()==digest,'Changed pinned source/config: '+path
    config='R1_partial_bce' if stage.startswith('SMOKE_') else stage
    config_path='CES_FL/proposals/upgrade_invariance_20261031_r1/configs/'+config+'.json'
    assert not (ROOT/stage).exists()
    child=None;lease=None;owner=ROOT.name+':'+stage+':ssh:'+str(os.getpid())
    def stop(number,_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,number)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic()
    try:
        while lease is None:
            occupied=set(call(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']).splitlines())
            inventory=call(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
            for line in sorted(inventory.splitlines(),key=lambda line:int(line.split(',')[2].strip()),reverse=True):
                index,uuid,free,util=[x.strip() for x in line.split(',')]
                if int(free)<24576:continue
                candidate=LOCKS/(host+'_'+index)
                try:candidate.mkdir()
                except FileExistsError:continue
                lease=candidate;(lease/'by').write_text(owner+'\n')
                current=set(call(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']).splitlines())
                check=call(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']); current_free={a.strip():int(b.strip()) for a,b in (line.split(',') for line in check.splitlines())};
                if current_free.get(uuid,0)<24576:
                    (lease/'by').unlink();lease.rmdir();lease=None;continue
                break
            if lease is None:
                write(ROOT/'runtime'/f'{stage}.binding.json',{'state':'WAITING_FOR_VRAM_HEADROOM','host':host,'execution_mode':'direct_ssh','time':time.time()})
                if time.monotonic()-started>7200:raise RuntimeError('No device with24GiB free within two hours; no existing process modified')
                time.sleep(20)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'CES_SOURCE_DIR':str(SOURCE),'CES_TRAIN_CONFIG':config_path,
            'CES_OUTPUT_DIR':str(ROOT/stage),'CES_SMOKE_ONLY':'1' if stage.startswith('SMOKE_') else '0',
            'CES_CUDA_MEMORY_FRACTION':'0.1','PYTHONPATH':str(ROOT/'cuda_policy')+':'+os.environ.get('PYTHONPATH',''),
            'PATH':str(Path(GIT).parent)+':/opt/pbs/bin:/usr/bin:/bin:'+os.environ.get('PATH',''),
            'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
        env.pop('PBS_JOBID',None)
        binding={'state':'BOUND','host':host,'gpu_uuid':uuid,'physical_index':int(index),'free_mib_before_launch':int(free),
            'execution_mode':'direct_ssh_bounded_shared_gpu','pbs_allocation':None,'cuda_allocator_fraction':0.1,'preexisting_compute_processes':list(occupied).count(uuid),'allocator_policy_sha256':hashlib.sha256((ROOT/'cuda_policy/sitecustomize.py').read_bytes()).hexdigest(),'worker_pid':os.getpid(),'time':time.time(),
            'source_commit':'875bce025084871438ef799ca9086985c4375672','config_sha256':hashlib.sha256((SOURCE/config_path).read_bytes()).hexdigest()}
        write(ROOT/'runtime'/f'{stage}.binding.json',binding)
        env['CES_EVAL_SMOKE_DEVICE']='cuda'
        command=[sys.executable,str(ROOT/'runtime/test_frozen_evaluation.py')] if stage.startswith('SMOKE_') else [sys.executable,'-u',str(ROOT/'runtime/evaluate_frozen.py'),'--protocol',str(ROOT/'protocol.json'),'--arm',stage,'--device','cuda']
        binding['operation']='software_inference_smoke' if stage.startswith('SMOKE_') else 'frozen_test_inference_only'
        binding['evaluation_protocol_sha256']=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
        binding['evaluation_driver_sha256']=hashlib.sha256((ROOT/'runtime/evaluate_frozen.py').read_bytes()).hexdigest()
        child=subprocess.Popen(command,cwd=SOURCE,env=env,start_new_session=True)
        write(ROOT/'runtime'/f'{stage}.process.json',{**binding,'launcher_pid':child.pid,'command':command,'stage':stage})
        code=child.wait()
        write(ROOT/'runtime'/f'{stage}.exit.json',{'exit_code':code,'host':host,'gpu_uuid':uuid,'worker_pid':os.getpid(),'elapsed_seconds':time.monotonic()-started,'finished_at':time.time(),'execution_mode':'direct_ssh'})
        raise SystemExit(code)
    except BaseException as error:
        if not isinstance(error,SystemExit) or error.code:
            write(ROOT/'runtime'/f'{stage}.failure.json',{'error':repr(error),'host':host,'traceback':traceback.format_exc(),'time':time.time()})
        raise
    finally:
        if lease and (lease/'by').exists() and (lease/'by').read_text().strip()==owner:
            (lease/'by').unlink();lease.rmdir()
if __name__=='__main__':main()
