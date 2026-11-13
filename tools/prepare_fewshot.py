"""从已清理的 b0 JSONL 生成论文的三种种子及嵌套小样本划分。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrs.adaptation import grouped_fewshot_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='RefSegRS 或 RRSIS-D 数据集根目录')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    inputs = [root/'b0'/f'{split}.jsonl' for split in ('train', 'val', 'test')]
    samples = [[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
               for path in inputs]
    for seed in (0, 1, 2):
        training, val = grouped_fewshot_splits(*samples, seed=seed)
        output = root/'b0'/'fewshot'/f'seed_{seed}'
        output.mkdir(parents=True, exist_ok=True)
        for name, subset in [(f'train_{k}', v) for k, v in training.items()] + [('val_20', val)]:
            (output/f'{name}.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in subset),encoding='utf-8')
        manifest = dict(seed=seed, train_budgets=[20,100,200], validation_budget=20,
                         source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                         selected_ids={str(k):[s.get('id',s['image']) for s in v] for k,v in training.items()},
                         validation_ids=[s.get('id',s['image']) for s in val], test_count=len(samples[2]))
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(str(output))


if __name__ == '__main__':
    main()
