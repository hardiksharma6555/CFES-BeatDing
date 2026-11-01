"""Software fixtures only: no new training or real test-set scoring."""
import csv, importlib.util, json, os, sys, tempfile, unittest, copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z')
spec=importlib.util.spec_from_file_location('frozen_eval_driver',ROOT/'runtime/evaluate_frozen.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
P=json.loads((ROOT/'protocol.json').read_text());UP,UNC=E.modules(P)
DEVICE=os.environ.get('CES_EVAL_SMOKE_DEVICE','cpu')
FIELDS=['image','label','identity_ids','source_ids','gender','race','age_bin','skin_bin','gender_race','ids8','cluster','family']
def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CES_EVALUATION_SOFTWARE_');self.root=Path(self.tmp.name)
        rng=np.random.default_rng(17);self.rows=[]
        for i in range(8):
            p=self.root/f'synthetic_{i}.png';Image.fromarray(rng.integers(0,256,(96,96,3),dtype=np.uint8)).save(p)
            self.rows.append(dict(image=str(p),label=str(i%2),identity_ids='synthetic'+str(i),source_ids='synthetic'+str(i),
                gender='fixture',race='fixture',age_bin='fixture',skin_bin='fixture',gender_race='fixture',ids8=str(i%2),cluster=str(i),family='synthetic'))
        self.unique=self.root/'unique.csv';csvwrite(self.unique,self.rows)
    def tearDown(self):self.tmp.cleanup()
    def test_native_duplicates_preserved_and_conflicts_rejected(self):
        native=self.root/'native.csv';csvwrite(native,self.rows+[self.rows[0],self.rows[3]])
        audit={'sha256':E.sha(native),'rows':10,'native_exact_duplicate_extra_rows':2,'unique_validation_index':str(self.unique),'unique_validation_index_sha256':E.sha(self.unique)}
        parsed=E.rows_for(UP.base,'dfdc',native,audit)
        self.assertEqual(len(parsed),10);self.assertEqual(parsed[0],parsed[8]);self.assertEqual(parsed[3],parsed[9])
        invalid=[dict(x) for x in self.rows]+[{**self.rows[0],'label':'1'}]
        csvwrite(native,invalid);audit.update(sha256=E.sha(native),rows=9,native_exact_duplicate_extra_rows=1)
        with self.assertRaises(AssertionError):E.rows_for(UP.base,'dfdc',native,audit)
    def test_changed_manifest_rejected(self):
        with self.assertRaises(AssertionError):E.rows_for(UP.base,'ffpp',self.unique,{'sha256':'0'*64,'rows':8,'native_exact_duplicate_extra_rows':0})
    def test_real_checkpoint_inference_matches_exporter_and_freezes_threshold(self):
        if DEVICE=='cuda':
            import torch
            self.assertTrue(torch.cuda.is_available())
        arm='R0_frozen_bce';entry=P['arms'][arm];model,ck=UP.load_model(entry['checkpoint'],DEVICE)
        rows=UP.base.read_rows(self.unique)
        protocol={'arms':{arm:entry},'manifest_audit':{'fixture':{'sha256':E.sha(self.unique),'native_exact_duplicate_extra_rows':0}},'bootstrap':{'seed':17,'replicates':20},'synthetic_smoke':True}
        ck['config']={**ck['config'],'workers':0}
        calibration=copy.deepcopy(ck['calibration']);before=E.sha(entry['checkpoint'])
        with patch.object(UP.base,'calibrate',side_effect=AssertionError('Test must not calibrate')):
            report=E.evaluate_dataset(UP,UNC,protocol,arm,model,ck,'fixture',rows,self.root/'wrapped')
            scores=np.load(self.root/'wrapped/predictions_private.npz')['logit']
            changed=[{**r,'label':1-r['label']} for r in rows]
            alternative=E.reports(UP.base,changed,scores,ck['calibration'],ck['config']['min_support'])
            self.assertEqual(alternative['calibration'],calibration)
            original=UP.evaluate(entry['checkpoint'],self.unique,self.root/'official',DEVICE)
        self.assertEqual(report['calibrated'],original['calibrated'])
        self.assertEqual(report['fixed_probability_0_5'],original['fixed_probability_0_5'])
        self.assertEqual(ck['calibration'],calibration);self.assertEqual(E.sha(entry['checkpoint']),before)
        self.assertTrue((self.root/'wrapped/DONE').exists())
if __name__=='__main__':unittest.main(verbosity=2)
