"""Fail-closed orchestration tests using explicitly synthetic metadata."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENTRY=Path(__file__).resolve().parents[1]/'scripts/verify_lite_run.py'
spec=importlib.util.spec_from_file_location('lite_completion',ENTRY)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


class CompletionGate(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='lite_completion_')
        self.root=Path(self.temp.name)
        self.run,self.prep,self.src=[self.root/n for n in ('run','prepared','src')]
        for p in (self.run,self.prep,self.src):p.mkdir()
        (self.src/'model.py').write_text('# synthetic controller fixture\n')
        (self.run/'best.pt').write_bytes(b'NOT_A_MODEL: synthetic controller fixture')
        (self.run/'student_fp32.pt').write_bytes(b'NOT_A_MODEL: synthetic export')
        self.config={'epochs':2,'arm':'bce','synthetic_fixture':True}
        self.history=[{'epoch':i,'originals_once_each':True,'unique_originals':16,
                       'validation':{'overall':{'auc':.75}}} for i in (1,2)]
        sources={'model.py':gate.sha(self.src/'model.py')}
        self.report={'status':'COMPLETED_VERIFIED','epochs_completed':2,'selected_epoch':1,
                     'synthetic_fixture':True,'validation':self.history[0]['validation'],
                     'source_hashes':sources,'selected_checkpoint_sha256':gate.sha(self.run/'best.pt'),
                     'fp32_file_bytes':(self.run/'student_fp32.pt').stat().st_size}
        self.write(self.prep/'config.json',self.config);self.write(self.run/'config.json',self.config)
        self.write(self.prep/'PREPARED.json',{'source_hashes':sources,'train_rows':16})
        self.write(self.run/'RUN_VERIFIED.json',self.report);self.write(self.run/'history.json',self.history)
        (self.run/'DONE').write_text('Synthetic fixture\n')

    def tearDown(self):self.temp.cleanup()
    def write(self,path,value):path.write_text(json.dumps(value))
    def check(self):return gate.verify(self.run,self.prep,self.src)

    def test_valid_complete_metadata(self):
        self.assertEqual(self.check()['selected_epoch'],1)

    def test_zero_exit_incomplete_stops_shell_chain(self):
        (self.run/'DONE').unlink()
        later=self.root/'DEPENDENT_ARM_MUST_NOT_START'
        # The preceding training process can succeed at the OS level while its
        # run is incomplete. The guard must still prevent the next operation.
        r=subprocess.run(['bash','-e','-c',
            'true\n"$1" "$2" --run "$3" --prepared "$4" --source-dir "$5"\ntouch "$6"',
            'test',sys.executable,str(ENTRY),str(self.run),str(self.prep),str(self.src),str(later)],
            capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertFalse(later.exists())
        self.assertIn('STOP_DEPENDENT_LAUNCH',r.stderr)

    def test_short_history_rejected_even_with_done(self):
        self.write(self.run/'history.json',self.history[:1])
        with self.assertRaisesRegex(ValueError,'epoch budget'):self.check()

    def test_tied_auc_must_select_earliest(self):
        self.report['selected_epoch']=2;self.write(self.run/'RUN_VERIFIED.json',self.report)
        with self.assertRaisesRegex(ValueError,'earliest'):self.check()

    def test_changed_checkpoint_rejected(self):
        (self.run/'best.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint bytes'):self.check()

    def test_changed_source_rejected(self):
        (self.src/'model.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'Current source'):self.check()

    def test_unresolved_budget_stop_rejected(self):
        self.write(self.run/'BUDGET_STOP.json',{'status':'STOPPED_MID_EPOCH'})
        with self.assertRaisesRegex(ValueError,'budget stop'):self.check()

    def test_changed_sampling_coverage_rejected(self):
        self.history[1]['unique_originals']=15;self.write(self.run/'history.json',self.history)
        with self.assertRaisesRegex(ValueError,'coverage'):self.check()


if __name__=='__main__':unittest.main()
