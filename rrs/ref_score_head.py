"""RefScoreHead: 可训练的指代相关性评分头 (本项目新增, 非官方迁移)。

给每个普通查询输出一个标量 logit, 学习"这个候选是否就是当前描述要找的目标"。
与现有模块的分工:
  - GroundingDINO token 分类头 (gd.class_embed): 查询 × 文本 token 的匹配 logits, 原 top1 依据;
  - 框预测头 (gd.bbox_embed) / MaskHead: 位置与像素, 与本头无关;
  - RefScoreHead: 单一标量, 监督来自各层 HungarianMatcher 已有的匹配结果 (见 criterion.py)。

六层 HED 共用同一份权重 (模型里只建一个 self.ref_score_head, 对六层分别调用),
输出原始 logits, 不在头内 sigmoid —— 损失用 with_logits 版本, 推理时才 sigmoid。
DN 查询在调用本头之前就已被剥离 (model.py 先 split 再评分)。
"""
import torch.nn as nn


class RefScoreHead(nn.Module):
    """(B, Q, D) -> (B, Q) 原始 logits。"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hs):
        # squeeze(-1) 而非无参数 squeeze(): B=1 或 Q=1 时不能把 batch/查询维一起消掉
        return self.net(hs).squeeze(-1)
