"""Synthetic software checks only. Set CES_UPGRADE_DEVICE=cuda for cluster smoke."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ces_fl_upgrade as up

DEVICE=os.environ.get('CES_UPGRADE_DEVICE','cpu')
torch.set_num_threads(2)


def fixture(root):
    """Every approval here applies only to synthetic software fixtures."""
    root=root.resolve()  # macOS /var is a symlink; manifests normalize paths.
    rng=np.random.default_rng(17); paths={}; pairs=[]; controls=[]
    for split in ('train','val'):
        rows=[]
        for i in range(4):
            path=root/f'{split}{i}.png'
            pixels=rng.integers(0,256,(80,80,3),dtype=np.uint8);Image.fromarray(pixels).save(path)
            row={'image':str(path),'label':str(i%2),'identity_ids':f'{split}{i}',
                'source_ids':f'{split}{i}','family':'synthetic',**{axis:str(i//2) for axis in up.base.AXES}}
            rows.append(row)
            if split=='train':
                control={'original_image':str(path),'original_sha256':up.base.sha(path),'edits':[]}
                for name in ('nochange_control','resize_control','condition_resize_control'):
                    pp=root/f'{split}{i}_{name}.png';Image.fromarray(pixels).save(pp)
                    control[name]=str(pp);control[name+'_sha256']=up.base.sha(pp)
                for j,direction in enumerate(('older','younger')):
                    pp=root/f'{split}{i}_{direction}.png';Image.fromarray(np.roll(pixels,j+1,axis=0)).save(pp)
                    pair={'pair_id':f'{i}_{direction}','original_image':str(path),'counterfactual_image':str(pp),
                        'kind':'demographic','attribute':'age','direction':direction,'approved':'1',
                        'identity_ids':row['identity_ids'],'source_ids':row['source_ids'],'spatial_validated':'0'}
                    pairs.append(pair);control['edits'].append({'pair_id':pair['pair_id'],
                        'counterfactual_image':str(pp),'independent_from_original':True,'sha256':up.base.sha(pp)})
                controls.append(control)
        manifest=root/(split+'.csv')
        with manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        paths[split]=str(manifest)
    pp=root/'pairs.csv'
    with pp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    cp=root/'controls.json';up.base.write(cp,controls)
    # Deliberately invalid as model weights: internal synthetic-run path never loads these.
    weights=root/'synthetic_weights';weights.write_text('synthetic fixture, not pretrained weights')
    checkpoint=root/'synthetic_checkpoint';checkpoint.write_text('synthetic fixture, not a model')
    c={'seed':17,'head_seed':17,'backbone':'xception','weights':str(weights),'hidden':8,'gate_hidden':4,
        'size':75,'epochs':1,'batch_size':2,'workers':0,'sampler':'balanced','balance_axis':'ids8',
        'lr':.001,'encoder_lr':.0001,'weight_decay':.0001,'target_fpr':.1,'min_support':1,
        **paths,'protocol_overlap_file':None,'encoder_checkpoint':str(checkpoint),
        'encoder_checkpoint_sha256':up.base.sha(checkpoint),'encoder_policy':'partial',
        'classification':'group_dro','dro_eta':.01,'dro_min_group_count':1,'consistency':'age',
        'pair_count':2,'minimum_pair_sources':4,'lambda_d':1.,'pairs':str(pp),'controls_manifest':str(cp),
        'purpose':'SYNTHETIC SOFTWARE FIXTURE ONLY','input_hashes':{}}
    c['input_hashes']={key:up.base.sha(c[key]) for key in ('train','val','weights','encoder_checkpoint','pairs','controls_manifest')}
    return c


class UpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DEVICE not in ('cpu','cuda'): raise ValueError('Smoke device must be cpu or cuda')
        if DEVICE=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA smoke requested but CUDA is unavailable; no GPU PASS')

    def test_partial_gradients_bn_and_decomposition_both_backbones(self):
        for backbone in ('xception','resnet50'):
            with self.subTest(backbone=backbone):
                up.base.seed_all(17)
                model=up.UpgradeCES(backbone,hidden=8,gate_hidden=4,pretrained=False).to(DEVICE)
                model.train();before={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
                opt=up.base.build_optimizer(model,{'lr':.001,'encoder_lr':.0001,'weight_decay':.0001})
                result=model(torch.randn(2,3,75,75,device=DEVICE))
                torch.testing.assert_close(result['logit'],result['global_contribution']+result['local_map'].mean((1,2))+result['fusion_bias'])
                loss=torch.nn.functional.binary_cross_entropy_with_logits(result['logit'],torch.tensor([0.,1.],device=DEVICE))
                loss.backward()
                selected=[p for p in model.encoder.parameters() if p.requires_grad]
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in selected),0)
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad))
                opt.step()
                changed=[]
                for name,value in model.encoder.state_dict().items():
                    if not torch.equal(value,before[name]):changed.append(name)
                    if any(tag in name for tag in ('running_mean','running_var','num_batches_tracked')):
                        self.assertTrue(torch.equal(value,before[name]),name)
                self.assertTrue(changed)
                self.assertTrue(all(any(k.startswith(n+'.') for n in model.selected_names) for k in changed))
                self.assertEqual([g['lr'] for g in opt.param_groups],[.0001,.001])

    def test_frozen_and_identical_initial_heads(self):
        a=up.UpgradeCES(pretrained=False,encoder_policy='frozen',hidden=8,gate_hidden=4)
        b=up.UpgradeCES(pretrained=False,encoder_policy='partial',hidden=8,gate_hidden=4)
        self.assertTrue(all(not p.requires_grad for p in a.encoder.parameters()))
        for key,value in a.state_dict().items():
            if not key.startswith('encoder.'):
                torch.testing.assert_close(value,b.state_dict()[key])

    def test_dro_numerical_gradient_missing_groups_and_reload(self):
        dro=up.GroupDRO(3,.2).to(DEVICE)
        losses=torch.tensor([.2,.8,1.5],device=DEVICE,requires_grad=True)
        value=dro(losses,torch.tensor([0,0,1],device=DEVICE))
        means=np.array([.5,1.5,0.]);q=np.exp(.2*means);q/=q.sum()
        self.assertAlmostEqual(float(value.detach()),float(q@means),places=6)
        value.backward()
        np.testing.assert_allclose(losses.grad.cpu().numpy(),[q[0]/2,q[0]/2,q[1]],rtol=1e-6)
        clone=up.GroupDRO(3,.2).to(DEVICE);clone.load_state_dict(dro.state_dict())
        torch.testing.assert_close(clone.log_q,dro.log_q)
        for _ in range(20):dro(torch.tensor([1000.],device=DEVICE),torch.tensor([2],device=DEVICE))
        self.assertTrue(torch.isfinite(dro.log_q).all())
        self.assertAlmostEqual(float(dro.log_q.exp().sum()),1.,places=5)

    def test_consistency_gradients_both_paths(self):
        original=torch.tensor([1.,-2.],device=DEVICE,requires_grad=True)
        edit=torch.tensor([.4,-.3],device=DEVICE,requires_grad=True)
        value=(torch.asinh(original)-torch.asinh(edit)).abs().mean();value.backward()
        self.assertTrue((original.grad.abs()>0).all());self.assertTrue((edit.grad.abs()>0).all())

    def test_encoder_transfer_preserves_heads_and_rejects_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();c=fixture(root)
            source=up.base.DiagramCES(pretrained=False,hidden=8,gate_hidden=4)
            target=up.UpgradeCES(pretrained=False,hidden=8,gate_hidden=4)
            heads={k:v.clone() for k,v in target.state_dict().items() if not k.startswith('encoder.')}
            checkpoint=root/'pretrain.pt'
            pc={**c,'stage':'pretrain','encoder_policy':'trainable','mode':'bce'}
            torch.save({'format':up.base.VERSION,'model':source.state_dict(),'config':pc,'epoch':1,
                'training_inputs':{'train_sha256':up.base.sha(c['train']),'val_sha256':up.base.sha(c['val'])}},checkpoint)
            config={**c,'encoder_checkpoint':str(checkpoint),'encoder_checkpoint_sha256':up.base.sha(checkpoint)}
            up.base.import_encoder(target,config)
            for key,value in source.encoder.state_dict().items():torch.testing.assert_close(value,target.encoder.state_dict()[key])
            for key,value in heads.items():torch.testing.assert_close(value,target.state_dict()[key])
            with self.assertRaises(ValueError):up.base.import_encoder(target,{**config,'encoder_checkpoint_sha256':'wrong'})
            with self.assertRaises(ValueError):up.base.import_encoder(target,{**config,'train':c['val']})

    def test_pair_schedule_balance_control_and_rng_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            c=fixture(Path(d));rows,_,_,_,_,a=up.preflight(c);_,_,_,_,_,b=up.preflight(c)
            ds=up.base.PairImages(rows,size=75)
            torch.manual_seed(30);state=torch.get_rng_state().clone()
            directions=Counter()
            for step in range(10):
                x,ed,chosen=a.load(ds,0,step,'age');xx,nc,other=b.load(ds,0,step,'nochange')
                self.assertEqual([(s,p['pair_id']) for s,p in chosen],[(s,p['pair_id']) for s,p in other])
                self.assertEqual(Counter(a.rows[s]['label'] for s,_ in chosen),{0:1,1:1})
                torch.testing.assert_close(x,xx);torch.testing.assert_close(xx,nc)
                self.assertFalse(torch.equal(x,ed));directions.update(p['direction'] for _,p in chosen)
            self.assertEqual(set(directions),{'older','younger'})
            self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_fail_closed_inputs_controls_and_pool_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            c=fixture(Path(d))
            with self.assertRaises(ValueError):up.preflight({**c,'minimum_pair_sources':300})
            with self.assertRaises(ValueError):up.validate_config({**c,'pair_count':3})
            with self.assertRaises(ValueError):up.validate_config({**c,'typo':True})
            with self.assertRaises(ValueError):up.validate_config({**c,'lambda_d':float('nan')})
            with self.assertRaises(ValueError):up.preflight({**c,'dro_min_group_count':20})
            controls=json.loads(Path(c['controls_manifest']).read_text())
            controls[0]['edits'][0]['independent_from_original']=False
            up.base.write(c['controls_manifest'],controls)
            with self.assertRaisesRegex(ValueError,'Changed input'):up.preflight(c)
            c['input_hashes']['controls_manifest']=up.base.sha(c['controls_manifest'])
            with self.assertRaisesRegex(ValueError,'chained'):up.preflight(c)

    def test_five_training_arms_reload_and_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=fixture(root)
            for name,policy,classification,consistency in [
                ('R0','frozen','bce','none'),('R1','partial','bce','none'),
                ('R2','partial','group_dro','none'),('RN','partial','group_dro','nochange'),
                ('RD','partial','group_dro','age')]:
                config={**c,'encoder_policy':policy,'classification':classification,'consistency':consistency,
                        'input_hashes':dict(c['input_hashes'])}
                if consistency=='none':
                    config.update(pair_count=0,lambda_d=0.,pairs=None,controls_manifest=None)
                    for k in ('pairs','controls_manifest'):config['input_hashes'].pop(k)
                checkpoint=up.run(config,root/name,DEVICE,_synthetic_smoke=True)
                m,ck=up.load_model(checkpoint,DEVICE)
                self.assertTrue(ck['synthetic_smoke']);self.assertIn('dro',ck)
                self.assertEqual(ck['input_hashes'],config['input_hashes'])
                record=json.loads((root/name/'history.jsonl').read_text())
                self.assertEqual(record['pair_count'],4 if consistency!='none' else 0)
                if consistency!='none':self.assertEqual(record['pair_labels'],{'0':2,'1':2})
                if name=='RD':
                    first=up.evaluate(checkpoint,c['val'],root/'eval1',DEVICE)
                    with Path(c['val']).open() as f:rows=list(csv.DictReader(f))
                    for row in rows:row['label']=str(1-int(row['label']))
                    flipped=root/'flipped.csv'
                    with flipped.open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                    second=up.evaluate(checkpoint,flipped,root/'eval2',DEVICE)
                    self.assertEqual(first['calibration'],second['calibration'])
                    self.assertEqual(first['fixed_probability_0_5']['threshold_logit'],0.)
                    self.assertEqual(first['threshold_source'],'selected checkpoint validation only')
                del m,ck


if __name__=='__main__':unittest.main()
