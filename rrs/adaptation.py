"""论文 §4.5.1：小样本划分和按实际参数更新次数驱动的两阶段适配。"""
from collections import defaultdict
from pathlib import PurePosixPath
import math
import random


def image_key(sample):
    return str(PurePosixPath(sample['image'].replace('\\', '/')))


def grouped_fewshot_splits(train, validation, test, seed, budgets=(20, 100, 200), val_budget=20):
    """每个原始图像只选一条完整指代表达，三档训练集嵌套且共用验证集。"""
    if seed not in (0, 1, 2):
        raise ValueError('论文抽样 seed 必须为 0、1 或 2')
    def grouped(samples):
        groups = defaultdict(list)
        for sample in samples:
            if not sample.get('caption', '').strip() or len(sample.get('objects', [])) != 1:
                raise ValueError('输入必须是已清理的单目标完整指代样本')
            groups[image_key(sample)].append(sample)
        return groups
    groups = [grouped(samples) for samples in (train, validation, test)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if set(groups[i]) & set(groups[j]):
            raise ValueError('train/val/test 的原始图像必须互不重叠')
    if len(groups[0]) < max(budgets) or len(groups[1]) < val_budget:
        raise ValueError('有效独立训练/验证图像不足，无法满足抽样预算')
    def select(group, count, rng):
        images = sorted(group)
        rng.shuffle(images)
        return [rng.choice(group[key]) for key in images[:count]]
    # 固定最大预算的一次抽样，防止为不同预算重新抽签而破坏包含关系。
    selected = select(groups[0], max(budgets), random.Random(seed))
    val = select(groups[1], val_budget, random.Random(seed + 1000003))
    return {budget: selected[:budget] for budget in budgets}, val


class AdaptationController:
    """每 50 次成功更新验证；阶段1最多300次更新，总预算1000次。

    checkpoint 的 best 使用原始 mIoU；停滞判定要求相对之前最佳提高至少
    0.1 个百分点。阶段2的学习率计数与六次停滞早停计数分别维护。
    """
    def __init__(self):
        self.stage = 1
        self.updates = 0
        self.best = -math.inf
        self.stagnant = 0
        self.lr_stagnant = 0
        self.stopped = False

    def updated(self):
        if self.stopped:
            raise RuntimeError('训练已停止')
        self.updates += 1
        return self.updates % 50 == 0

    def observe(self, miou):
        if self.updates == 0 or self.updates % 50:
            raise ValueError('只能在每50次成功参数更新后提交验证指标')
        if not math.isfinite(miou):
            raise ValueError('验证 mIoU 必须有限')
        improved = miou > self.best
        significant = miou >= self.best + 0.1
        self.best = max(self.best, miou)
        self.stagnant = 0 if significant else self.stagnant + 1
        self.lr_stagnant = 0 if significant else self.lr_stagnant + 1
        event = dict(save_best=improved, transition=False, decay=False, stop=False)
        if self.stage == 1 and (self.stagnant >= 3 or self.updates >= 300):
            self.stage = 2
            self.stagnant = self.lr_stagnant = 0
            event['transition'] = True
        elif self.stage == 2:
            if self.stagnant >= 6:
                self.stopped = event['stop'] = True
            elif self.lr_stagnant >= 3:
                event['decay'] = True
                self.lr_stagnant = 0
        if self.updates >= 1000:
            self.stopped = event['stop'] = True
        return event
