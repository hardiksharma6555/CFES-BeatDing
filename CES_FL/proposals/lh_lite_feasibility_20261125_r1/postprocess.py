"""CPU-only INT8 validation and paired uncertainty after completed training."""
import argparse
import json
import os
from pathlib import Path
import sys

# Set before importing torch; this job must not allocate a GPU.
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
import ces_fl_lite_training as tr
import numpy as np
import torch


def exported_float(run, quantized, out, fixture=False):
    """Rebuild/evaluate the actual FP32 CPU artifact for a matched INT8 comparison."""
    model,ck=tr.load_student(run/'best.pt','cpu',fixture)
    cache=json.loads((quantized/'validation_scores_PRIVATE.json').read_text())
    c=ck['config'];teacher_cache=Path(c['teacher_run'])/'validation_scores_PRIVATE.json'
    if cache.get('split')!='validation' or cache['manifest_sha256']!=c['val_sha256']:
        raise ValueError('FP32 deployment comparison requires the bound VAL cohort')
    if tr.native.sha(teacher_cache)!=c['teacher_cache_sha256']:
        raise ValueError('Bound teacher VAL cache changed')
    expected=json.loads(teacher_cache.read_text())
    if cache['rows']!=expected['rows']:raise ValueError('FP32/INT8 cohort differs from native teacher VAL')
    rows=cache['rows'];base=ck['native_config'];out=tr.native.fresh_dir(out)
    artifact=out/'student_fp32.pt'
    torch.jit.save(torch.jit.freeze(torch.jit.script(model.eval())),str(artifact))
    loaded=torch.jit.load(str(artifact)).eval()
    z=tr.native.predict(loaded,rows,base,torch.device('cpu'))
    cal=tr.native.calibrate(z[np.array([r['label']==0 for r in rows])],base['target_fpr'])
    tr.native.write_json(out/'EXPORT_REPORT.json',{'format':'FP32_TORCHSCRIPT_CPU',
        'model_sha256':tr.native.sha(artifact),'model_bytes':artifact.stat().st_size,
        'under_10_decimal_MB':artifact.stat().st_size<10000000,
        'validation':tr.native.metric_report(rows,z,cal['threshold_logit'],base['min_eval_support']),
        'calibration':cal,'source_hashes':tr.source_hashes(),'test_read':False,'synthetic_fixture':fixture})
    tr.native.write_json(out/'validation_scores_PRIVATE.json',{
        'rows':rows,'logits':z.tolist(),'threshold':cal['threshold_logit'],
        'manifest_sha256':cache['manifest_sha256'],'checkpoint_sha256':tr.native.sha(artifact),
        'min_support':base['min_eval_support'],'split':'validation','test_read':False})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',required=True);p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    prepared,runs=Path(a.prepared),Path(a.runs)
    if not (runs/'GPU_STAGE_DONE').is_file():raise ValueError('GPU stage is incomplete; preserve existing runs')
    for arm in tr.ARMS:
        r=json.loads((runs/arm/'RUN_VERIFIED.json').read_text())
        if r['status']!='COMPLETED_VERIFIED':raise ValueError(f'Unverified {arm}')
    out=tr.native.fresh_dir(a.out)
    for arm in tr.ARMS:
        prep=prepared/('prepared_'+arm);run=runs/arm
        c=json.loads((prep/'config.json').read_text())
        q=out/(arm+'_int8')
        tr.export_int8(prep,run,q)
        fp=out/(arm+'_fp32_cpu');exported_float(run,q,fp)
        teacher=Path(c['teacher_run'])/'validation_scores_PRIVATE.json'
        tr.compare(fp/'validation_scores_PRIVATE.json',teacher,out/(arm+'_float_minus_teacher.json'))
        tr.compare(q/'validation_scores_PRIVATE.json',teacher,out/(arm+'_int8_minus_teacher.json'))
        tr.compare(q/'validation_scores_PRIVATE.json',fp/'validation_scores_PRIVATE.json',out/(arm+'_int8_minus_float.json'))
        print(arm,'CPU quantization and paired validation analysis complete',flush=True)
    tr.compare(out/'fair_response_int8/validation_scores_PRIVATE.json',
               out/'kd_int8/validation_scores_PRIVATE.json',out/'fair_int8_minus_kd_int8.json')
    tr.compare(out/'fair_response_int8/validation_scores_PRIVATE.json',
               out/'bce_int8/validation_scores_PRIVATE.json',out/'fair_int8_minus_bce_int8.json')
    (out/'DONE').write_text('CPU stage complete; contributor-bootstrap support and scientific conclusions require review.\n')


if __name__=='__main__':main()
