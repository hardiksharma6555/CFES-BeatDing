#!/usr/bin/env python3
"""Reject incomplete student runs before launching dependent work.

This is an orchestration/provenance check, not an independent proof of model
accuracy. In particular a process exit code zero is not a completion receipt.
Uses the standard library only; it never initializes CUDA or loads checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def verify(run, prepared, source_dir):
    run,prepared,source_dir=Path(run),Path(prepared),Path(source_dir)
    for name in ('DONE','RUN_VERIFIED.json','config.json','history.json','best.pt','student_fp32.pt'):
        if not (run/name).is_file():raise ValueError(f'Incomplete run: missing {name}')
    if (run/'BUDGET_STOP.json').exists():raise ValueError('Run has an unresolved budget stop')
    expected=json.loads((prepared/'config.json').read_text())
    preparation=json.loads((prepared/'PREPARED.json').read_text())
    config=json.loads((run/'config.json').read_text())
    report=json.loads((run/'RUN_VERIFIED.json').read_text())
    history=json.loads((run/'history.json').read_text())
    if config!=expected:raise ValueError('Completed run differs from prepared configuration')
    if report.get('status')!='COMPLETED_VERIFIED':raise ValueError('Run is not COMPLETED_VERIFIED')
    if report.get('synthetic_fixture')!=config.get('synthetic_fixture'):
        raise ValueError('Fixture/production scope differs')
    epochs=config['epochs']
    if report.get('epochs_completed')!=epochs or len(history)!=epochs:
        raise ValueError('Requested epoch budget was not completed')
    if [r['epoch'] for r in history]!=list(range(1,epochs+1)):
        raise ValueError('Non-contiguous epoch history')
    if not all(r.get('originals_once_each') is True and r.get('unique_originals')==preparation['train_rows'] for r in history):
        raise ValueError('Original-image coverage differs')
    best=max(history,key=lambda r:r['validation']['overall']['auc'])
    if report.get('selected_epoch')!=best['epoch'] or report.get('validation')!=best['validation']:
        raise ValueError('Selected checkpoint differs from earliest maximum validation AUC')
    if sha(run/'best.pt')!=report.get('selected_checkpoint_sha256'):
        raise ValueError('Selected checkpoint bytes changed')
    if (run/'student_fp32.pt').stat().st_size!=report.get('fp32_file_bytes'):
        raise ValueError('Exported FP32 artifact size changed')
    sources=preparation['source_hashes']
    if report.get('source_hashes')!=sources:raise ValueError('Completion source differs from preparation')
    if not sources:raise ValueError('Missing source hashes')
    for name,expected_hash in sources.items():
        if Path(name).name!=name:raise ValueError('Source receipt must contain basenames')
        if sha(source_dir/name)!=expected_hash:raise ValueError('Current source differs: '+name)
    return {'status':'VERIFIED_COMPLETE_FOR_DEPENDENT_STAGE','arm':config['arm'],
            'epochs_completed':epochs,'selected_epoch':best['epoch'],
            'checkpoint_sha256':report['selected_checkpoint_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--source-dir',required=True);a=p.parse_args()
    try:result=verify(a.run,a.prepared,a.source_dir)
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:
        p.exit(2,f'STOP_DEPENDENT_LAUNCH: {e}\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
