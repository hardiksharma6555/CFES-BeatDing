"""Bounded private CPU preparation/backward checks; never trains or reads TEST."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
import ces_fl_lite_training as tr

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=tr.native.fresh_dir(a.out)
tr.native.write_json(out/'PROCESS.json',{'pid':os.getpid(),'started_unix':time.time(),
                                        'source_hashes':tr.source_hashes(),'cpu_only':True})
results={}
try:
    for arm in tr.ARMS:
        c=json.loads((HERE/'configs'/(arm+'.json')).read_text())
        result=tr.prepare(c,out/('prepared_'+arm))
        results[arm]={'prepare':result}
        print(json.dumps({'arm':arm,'phase':'prepare','status':result['status']}),flush=True)
    result=tr.dry_run(out/'prepared_fair_response',out/'backward_fair_response','cpu')
    results['fair_response']['backward']=result
    tr.native.write_json(out/'CPU_PRODUCTION_CHECKS.json',{'status':'PASS','results':results,
        'production_optimizer_updates':0,'validation_inference':False,'test_read':False})
    (out/'DONE').write_text('0\n')
except Exception as e:
    tr.native.write_json(out/'FAILURE.json',{'type':type(e).__name__,'error':str(e),'completed_checks':results})
    (out/'DONE').write_text('1\n')
    raise
