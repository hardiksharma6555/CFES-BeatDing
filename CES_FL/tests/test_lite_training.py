"""Synthetic end-to-end controls; no real dataset or task-performance claims."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ces_fl_lite_training as tr
native=tr.native


def fixture(root):
    torch.set_num_threads(2);native.seed_all(17)
    paths={}
    for split in ('train','val'):
        rows=[]
        for g in range(4):
            for label in (0,1):
                for rep in range(2):
                    stem=f'{split}_{g}_{label}_{rep}';p=root/(stem+'.png')
                    a=np.random.default_rng(len(rows)+(0 if split=='train' else 100)).integers(0,256,(40,40,3),dtype=np.uint8)
                    Image.fromarray(a).save(p)
                    gender=['Female','Male'][g//2];race=['A','B'][g%2]
                    rows.append({'image':str(p),'label':label,'identity_ids':stem,'source_ids':stem,
                                 'gender':gender,'race':race,'gender_race':gender+'-'+race,
                                 'age_bin':str(g//2),'skin_bin':str(g%2),'ids8':str(g),'family':'synthetic'})
        paths[split]=root/(split+'.csv')
        with paths[split].open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tc={**native.defaults(),'dataset_id':'SYNTHETIC_SOFTWARE_FIXTURE','train':str(paths['train']),
        'val':str(paths['val']),'train_sha256':native.sha(paths['train']),'val_sha256':native.sha(paths['val']),
        'size':71,'epochs':1,'batch_size':4,'workers':0,'min_train_support':1,'min_eval_support':1,
        'train_calibration_size':8,'group_per_class':1,'aux_every':1,'axes':['gender_race'],
        'encoder_policy':'frozen','error_control':'hierarchical_real','amp':False,
        'synthetic_fixture':True,'source_sha256':tr.NATIVE_SHA}
    pre=root/'native_preflight';native.preflight(tc,pre)
    rows,val,_=native.checked_inputs(tc,pre)
    teacher=native.Detector(policy='frozen').eval()
    z=native.predict(teacher,val,tc,torch.device('cpu'));cal=native.calibrate(z[np.array([r['label']==0 for r in val])],.1)
    report=native.metric_report(val,z,cal['threshold_logit'],1)
    run=root/'teacher';run.mkdir()
    ck={'format':native.VERSION,'model':teacher.state_dict(),'config':tc,'epoch':1,
        'source_sha256':tr.NATIVE_SHA,'synthetic_fixture':True,'validation':report,'calibration':cal,
        'qualification_sha256':None}
    torch.save(ck,run/'best.pt');native.write_json(run/'config.json',tc)
    native.write_json(run/'history.json',[{'epoch':1,'validation':report}])
    native.write_json(run/'RUN_VERIFIED.json',{'status':'COMPLETED_VERIFIED','source_sha256':tr.NATIVE_SHA,
        'synthetic_fixture':True,'config_digest':native.digest(tc),'epochs_completed':1,'selected_epoch':1,
        'selected_checkpoint_sha256':native.sha(run/'best.pt'),'validation':report})
    (run/'DONE').write_text('Synthetic controller fixture; teacher initialized, no task-training claim.\n')
    native.write_json(run/'validation_scores_PRIVATE.json',{'rows':val,'logits':z.tolist(),'threshold':cal['threshold_logit'],
        'split':'validation','checkpoint_sha256':native.sha(run/'best.pt'),'manifest_sha256':tc['val_sha256'],'min_support':1})
    c={'format':tr.VERSION,'dataset_id':tc['dataset_id'],'student':'micro_ds_096','student_init':None,
       'student_init_sha256':None,'teacher_run':str(run),'teacher_sha256':native.sha(run/'best.pt'),
       'teacher_cache_sha256':native.sha(run/'validation_scores_PRIVATE.json'),'teacher_preflight':str(pre),
       'qualification':None,'train':tc['train'],'val':tc['val'],'train_sha256':tc['train_sha256'],
       'val_sha256':tc['val_sha256'],'overlap_file':None,'overlap_sha256':None,
       'seed':17,'epochs':2,'batch_size':4,'workers':0,'lr':.0001,'weight_decay':.0001,
       'lambda_kd':.5,'lambda_response_kd':.1,'kd_temperature':2.,'arm':'bce',
       'max_train_seconds':600.,'quant_calibration_size':8,'synthetic_fixture':True}
    return c


class LiteLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='cesfl_lite_lifecycle_')
        cls.root=Path(cls.temp.name);cls.config=fixture(cls.root)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_production_cannot_use_synthetic_fixture(self):
        with self.assertRaisesRegex(ValueError,'fixture'):tr.check_config(self.config)

    def test_calibration_retains_class_group_coverage(self):
        rows=native.read_manifest(self.config['train'])
        chosen=tr.calibration_indices(rows,8,17)
        self.assertEqual(len(set(chosen)),8)
        self.assertEqual({(rows[i]['label'],rows[i]['ids8']) for i in chosen},
                         {(r['label'],r['ids8']) for r in rows})
        with self.assertRaises(ValueError):tr.calibration_indices(rows,7,17)

    def test_full_three_arm_lifecycle_and_actual_quantized_export(self):
        for arm in tr.ARMS:
            with self.subTest(arm=arm):
                c={**self.config,'arm':arm};prep=self.root/('prepare_'+arm);run=self.root/('run_'+arm)
                tr.prepare(c,prep,fixture=True)
                result=tr.train(prep,run,'cpu',fixture=True)
                self.assertEqual(result['status'],'COMPLETED_VERIFIED')
                self.assertEqual(result['epochs_completed'],2)
                history=json.loads((run/'history.json').read_text())
                best=max(history,key=lambda r:r['validation']['overall']['auc'])
                self.assertEqual(result['selected_epoch'],best['epoch'])
                self.assertEqual(history[-1]['unique_originals'],16)
                self.assertEqual(history[-1]['real_source_pairs'],8 if arm=='fair_response' else 0)
                self.assertEqual(native.sha(Path(c['teacher_run'])/'best.pt'),c['teacher_sha256'])
                if arm=='fair_response':
                    self.assertGreater(history[0]['losses_step_mean']['teacher_response_kd'],0)
                    ex=tr.export_int8(prep,run,self.root/'quant','qnnpack',fixture=True)
                    self.assertTrue(ex['under_10_decimal_MB']);self.assertFalse(ex['esp32_compatible_artifact'])
                    self.assertEqual(ex['calibration_split'],'TRAIN')
                    self.assertEqual(ex['validation']['overall']['n'],16)
                    path=Path(__file__).resolve().parents[1]/'proposals/lh_lite_feasibility_20261125_r1/postprocess.py'
                    spec=importlib.util.spec_from_file_location('lite_postprocess',path)
                    post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
                    post.exported_float(run,self.root/'quant',self.root/'fp32_cpu',fixture=True)
                    fp=json.loads((self.root/'fp32_cpu/EXPORT_REPORT.json').read_text())
                    self.assertTrue(fp['under_10_decimal_MB']);self.assertTrue(fp['synthetic_fixture'])
                    self.assertEqual(fp['validation']['overall']['n'],16)
                    tr.compare(self.root/'quant/validation_scores_PRIVATE.json',
                               Path(c['teacher_run'])/'validation_scores_PRIVATE.json',
                               self.root/'paired.json',replicates=20)
                    self.assertEqual(json.loads((self.root/'paired.json').read_text())['replicates'],20)

    def test_epoch_boundary_resume_matches_uninterrupted(self):
        c={**self.config,'arm':'kd'};prep=self.root/'resume_prepared'
        tr.prepare(c,prep,fixture=True)
        a=self.root/'uninterrupted';b=self.root/'resumed'
        tr.train(prep,a,'cpu',fixture=True)
        stop=tr.train(prep,b,'cpu',fixture=True,stop_after=1)
        self.assertEqual(stop['status'],'INCOMPLETE_EPOCH_BOUNDARY_STOP')
        tr.train(prep,b,'cpu',resume=True,fixture=True)
        ac=torch.load(a/'last.pt',weights_only=True);bc=torch.load(b/'last.pt',weights_only=True)
        for k in ac['model']:torch.testing.assert_close(ac['model'][k],bc['model'][k],atol=0,rtol=0)
        self.assertEqual(ac['validation'],bc['validation'])

    def test_changed_cache_rejected_and_bn_buffers_frozen(self):
        c={**self.config,'teacher_cache_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'cache changed'):tr.teacher_evidence(c,fixture=True)
        m=tr.new_student(self.config,fixture=True);m.train()
        before={k:v.clone() for k,v in m.named_buffers() if 'running_' in k or 'num_batches_tracked' in k}
        m(torch.rand(4,3,71,71))['logit'].sum().backward()
        after=dict(m.named_buffers())
        for k,v in before.items():self.assertTrue(torch.equal(v,after[k]))

    def test_dry_run_and_budget_stop(self):
        c={**self.config,'arm':'fair_response'};prep=self.root/'dry_prepared'
        tr.prepare(c,prep,fixture=True)
        dry=tr.dry_run(prep,self.root/'dry','cpu',fixture=True)
        self.assertEqual(dry['optimizer_updates'],0)
        self.assertEqual(dry['native_auxiliary_axis'],'gender_race')
        c={**self.config,'max_train_seconds':.01};prep=self.root/'budget_prepared'
        tr.prepare(c,prep,fixture=True)
        run=self.root/'budget_run'
        try:
            result=tr.train(prep,run,'cpu',fixture=True)
            self.assertEqual(result['status'],'INCOMPLETE_EPOCH_BOUNDARY_STOP')
        except RuntimeError as e:
            self.assertIn('budget exhausted',str(e))
            self.assertTrue((run/'BUDGET_STOP.json').is_file())
            self.assertFalse((run/'DONE').exists())


if __name__=='__main__':unittest.main()
