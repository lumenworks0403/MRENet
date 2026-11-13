"""小样本预算、图像隔离、阶段切换、早停及解冻优化器状态检查。"""
import unittest
import torch
from torch import nn

from rrs.adaptation import grouped_fewshot_splits, AdaptationController, image_key
from tools.finetune import adaptation_optimizer, enter_second_stage


def samples(prefix, count, expressions=1):
    return [dict(id=f'{prefix}-{i}-{j}',image=f'{prefix}/{i}.png',caption=f'the object {j}',
                  objects=[dict(bbox=[0,0,2,2],segmentation={})])
             for i in range(count) for j in range(expressions)]


class TestFewshotSplits(unittest.TestCase):
    def test_nested_budgets_one_expression_and_reproducible(self):
        source = (samples('train',220,3),samples('val',25,2),samples('test',30))
        for seed in (0,1,2):
            train, val = grouped_fewshot_splits(*source,seed=seed)
            again, val_again = grouped_fewshot_splits(*source,seed=seed)
            self.assertEqual(train,again)
            self.assertEqual(val,val_again)
            self.assertEqual(train[20],train[100][:20])
            self.assertEqual(train[100],train[200][:100])
            self.assertEqual(len({image_key(s) for s in train[200]}),200)
            self.assertEqual(len({image_key(s) for s in val}),20)
        a,_ = grouped_fewshot_splits(*source,seed=0)
        b,_ = grouped_fewshot_splits(*source,seed=1)
        self.assertNotEqual(a[20],b[20])

    def test_overlap_and_insufficient_unique_images_rejected(self):
        train = samples('train',220)
        with self.assertRaises(ValueError):
            grouped_fewshot_splits(train,train[:25],samples('test',20),seed=0)
        with self.assertRaises(ValueError):
            grouped_fewshot_splits(samples('train',100,10),samples('val',25),samples('test',20),seed=0)


class TestAdaptationController(unittest.TestCase):
    @staticmethod
    def observe(controller, score):
        for offset in range(50):
            due=controller.updated()
            assert due == (offset == 49)
        return controller.observe(score)

    def test_first_plateau_switches_stage_and_second_stops_after_six(self):
        c=AdaptationController()
        self.assertTrue(self.observe(c,10.)['save_best'])
        for score in (10.02,10.04):
            self.assertFalse(self.observe(c,score)['transition'])
        event=self.observe(c,10.06)
        self.assertTrue(event['transition'])
        self.assertEqual(c.stage,2)
        self.assertEqual(c.stagnant,0)
        for index in range(1,7):
            event=self.observe(c,9.)
            self.assertEqual(event['decay'],index==3)
            self.assertEqual(event['stop'],index==6)
        self.assertTrue(c.stopped)

    def test_stage_one_cap_and_total_update_budget(self):
        c=AdaptationController()
        for index in range(1,21):
            event=self.observe(c,float(index))
            self.assertEqual(event['transition'],index==6)
            self.assertEqual(event['stop'],index==20)
        self.assertEqual(c.updates,1000)

    def test_significant_improvement_resets_stagnation(self):
        c=AdaptationController()
        self.observe(c,10.)
        self.observe(c,9.)
        self.observe(c,9.)
        self.assertFalse(self.observe(c,10.2)['transition'])
        self.assertEqual(c.stagnant,0)

    def test_backbone_unfreezes_with_task_moments_preserved(self):
        model=nn.Module()
        model.gd=nn.Module()
        model.gd.bert=nn.Linear(2,2)
        model.gd.backbone=nn.Sequential(nn.Linear(2,2))
        model.task=nn.Linear(2,1)
        optimizer=adaptation_optimizer(model,{})
        self.assertFalse(model.gd.backbone[0].weight.requires_grad)
        self.assertFalse(model.gd.bert.weight.requires_grad)
        loss=model.task(model.gd.backbone(torch.ones(1,2))).square().sum()
        loss.backward()
        optimizer.step()
        moment=optimizer.state[model.task.weight]['exp_avg'].clone()
        enter_second_stage(optimizer)
        self.assertTrue(model.gd.backbone[0].weight.requires_grad)
        self.assertFalse(model.gd.bert.weight.requires_grad)
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[1e-5,1e-6])
        torch.testing.assert_close(moment,optimizer.state[model.task.weight]['exp_avg'])


if __name__=='__main__':
    unittest.main(verbosity=2)
