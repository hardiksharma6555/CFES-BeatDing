"""Research probe/audit software tests. No real data or detector training."""
import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

import audit_checkpoint as audit
import mechanism_probe as probe

DEVICE = os.environ.get('CES_AUDIT_DEVICE', 'cpu')
torch.set_num_threads(2)


class MechanismTests(unittest.TestCase):
    def test_constructed_failure_and_gradients(self):
        result = probe.probe()
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['real_fairness_improvement_verified'])

    def test_independent_numpy_reference_unequal_cell_sizes(self):
        rng = np.random.default_rng(7)
        e = rng.normal(size=(14, 2))
        y = np.array([0]*5 + [1]*9)
        g = np.array([0]*2 + [1]*3 + [0]*4 + [1]*5)
        expected = 0.
        for label in (0, 1):
            sub = e[y == label]; gs = g[y == label]
            kernel = np.exp(-((sub[:, None]-sub[None, :])**2).sum(-1)/32.)
            ref = np.array([1/(2*np.sum(gs == group)) for group in gs])
            for group in (0, 1):
                w = (gs == group)/np.sum(gs == group)-ref
                expected += float(w @ kernel @ w)/4
        actual = probe.conditional_joint_loss(torch.tensor(e), torch.tensor(y), torch.tensor(g))
        self.assertAlmostEqual(float(actual), expected, places=12)

    def test_order_and_class_shift_invariance(self):
        e, y, g = probe.fixture(torch.tensor(3., dtype=torch.float64))
        original = probe.conditional_joint_loss(e, y, g)
        shifted = e + (y.double()*100)[:, None]
        perm = torch.randperm(len(e), generator=torch.Generator().manual_seed(99))
        torch.testing.assert_close(original, probe.conditional_joint_loss(shifted[perm], y[perm], g[perm]))

    def test_class_separation_does_not_get_aligned(self):
        e, y, g = probe.fixture(torch.tensor(0., dtype=torch.float64))
        e = e + (y.double()*1000)[:, None]
        self.assertLess(abs(float(probe.conditional_joint_loss(e, y, g))), 1e-12)
        with self.assertRaises(ValueError):
            probe.conditional_joint_loss(e[y == 0], y[y == 0], g[y == 0])

    def test_summary_and_frozen_threshold(self):
        e, y, g = probe.fixture(torch.tensor(3., dtype=torch.float64))
        rows = [{'label':int(y[i]), 'gender_race':str(int(g[i])), 'cluster':str(i)} for i in range(len(y))]
        c = {'axis':'gender_race','min_support':2,'kernel_samples_per_cell':4,
             'seed':17,'kernel_sigma':4.,'gain_perturbations':[-.1,0.,.1]}
        report = audit.summarize(rows, e[:,0].numpy(), e[:,1].numpy(), 0., 0., c)
        self.assertEqual({r['cancellation_amount_logit_units'] for r in report['group_class_means']}, {6.})
        self.assertTrue(all(r['threshold_logit'] == 0 for r in report['gain_stress']))
        self.assertEqual(report['gain_stress'][1]['axis']['fpr_gap'], 0.)
        self.assertEqual(report['gain_stress'][2]['axis']['fpr_gap'], .5)
        with self.assertRaises(ValueError):
            audit.summarize(rows, [float('nan')]*len(rows), e[:,1].numpy(), 0, 0, c)

    def test_audit_end_to_end_checkpoint_hashes_and_read_only(self):
        if DEVICE not in ('cpu','cuda') or (DEVICE == 'cuda' and not torch.cuda.is_available()):
            raise RuntimeError('Requested smoke device unavailable; no CUDA pass')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); rows = []; rng = np.random.default_rng(17)
            for i in range(8):
                image = root/f'{i}.png'
                Image.fromarray(rng.integers(0,256,(75,75,3),dtype=np.uint8)).save(image)
                rows.append({'image':str(image),'label':i%2,'identity_ids':str(i),
                             'source_ids':str(i),'gender_race':str(i//4),'cluster':str(i)})
            manifest = root/'val.csv'
            with manifest.open('w',newline='') as f:
                w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            audit.up.base.seed_all(17)
            model = audit.up.UpgradeCES(pretrained=False,hidden=8,gate_hidden=4).to(DEVICE).eval()
            loaded_rows = audit.up.base.read_rows(manifest)
            z = audit.up.base.predict(model,loaded_rows,DEVICE,batch_size=2,size=75)
            cal = audit.up.base.calibrate(z[np.array([r['label'] for r in loaded_rows]) == 0], .1)
            ck = {'format':audit.up.VERSION,'model':model.state_dict(),'epoch':1,
                  'config':{'backbone':'xception','hidden':8,'gate_hidden':4,
                            'encoder_policy':'partial','head_seed':17,'size':75},
                  'input_hashes':{'val':audit.up.base.sha(manifest),'train':'NOT_VAL'},
                  'source_sha256':{'upgrade':audit.UPGRADE_SHA,'base':audit.up.BASE_SHA},
                  'calibration':cal,'synthetic_smoke':True,
                  'validation':audit.up.base.metrics(loaded_rows,z,cal['threshold_logit'],2)}
            checkpoint = root/'synthetic.pt'; torch.save(ck,checkpoint)
            c = {'checkpoint':str(checkpoint),'checkpoint_sha256':audit.up.base.sha(checkpoint),
                 'manifest':str(manifest),'manifest_sha256':audit.up.base.sha(manifest),
                 'split':'val','axis':'gender_race','min_support':2,'batch_size':2,'workers':0,
                 'seed':17,'kernel_sigma':4.,'kernel_samples_per_cell':2,
                 'gain_perturbations':[-.1,0.,.1]}
            before = {k:v.clone() for k,v in model.state_dict().items()}
            result = audit.run(c,root/'audit',DEVICE)
            self.assertFalse(result['training_started']); self.assertFalse(result['test_read'])
            self.assertTrue(result['synthetic_smoke'])
            self.assertEqual(result['calibration'],cal)
            self.assertTrue(result['validation_auc_replay']['matches_at_1e_minus_5'])
            self.assertEqual(result['gain_stress'][1]['overall'], result['nominal_metrics']['overall'])
            self.assertEqual(audit.up.base.sha(checkpoint),c['checkpoint_sha256'])
            for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
            with self.assertRaises(FileExistsError): audit.run(c,root/'audit',DEVICE)
            with self.assertRaises(ValueError): audit.run({**c,'checkpoint_sha256':'wrong'},root/'bad',DEVICE)
            with self.assertRaises(ValueError): audit.run({**c,'split':'train'},root/'bad',DEVICE)
            for override in ({'split':'test'}, {'min_support':0}, {'kernel_sigma':0},
                             {'gain_perturbations':[.1]}, {'unexpected':True}):
                with self.assertRaises(ValueError): audit.validate_config({**c,**override})


if __name__ == '__main__':
    unittest.main()
