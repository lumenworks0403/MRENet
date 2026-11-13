"""论文 §4.5.1 的两阶段小样本迁移；从 RIS-LAD 最佳权重开始独立适配。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rrs.adaptation import AdaptationController, image_key
from rrs.config import load_cfg
from rrs.criterion import build_criterion
from rrs.data import RefSegDataset, collate_fn, load_samples
from rrs.model import build_model, load_model_state
from rrs.model_mode import resolve_model_mode
from rrs.training import build_grad_scaler, to_device
from tools.train import evaluate


def adaptation_optimizer(model, cfg):
    """冻结 BERT 和骨干；骨干参数提前入组，解冻时保留其他参数的 AdamW 动量。"""
    task, visual = [], []
    for name, parameter in model.named_parameters():
        if name.startswith('gd.bert.'):
            parameter.requires_grad_(False)
        elif name.startswith('gd.backbone.0.'):
            parameter.requires_grad_(False)
            visual.append(parameter)
        elif parameter.requires_grad:
            task.append(parameter)
    return torch.optim.AdamW([dict(params=task,lr=1e-4), dict(params=visual,lr=1e-6)],
                             weight_decay=1e-4)


def enter_second_stage(optimizer):
    optimizer.param_groups[0]['lr'] = 1e-5
    optimizer.param_groups[1]['lr'] = 1e-6
    for parameter in optimizer.param_groups[1]['params']:
        parameter.requires_grad_(True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-ckpt', required=True, help='RIS-LAD 验证集选出的模型权重')
    parser.add_argument('--shots', type=int, choices=(20,100,200), default=100)
    parser.add_argument('--seed', type=int, choices=(0,1,2), default=0)
    parser.add_argument('--opts', nargs='*')
    parser.add_argument('--eval-test', action='store_true', help='训练结束后仅对选出的最佳权重测试一次')
    args = parser.parse_args()
    cfg = resolve_model_mode(load_cfg(args.config,args.opts), opts=args.opts)
    if cfg['batch_size'] != 2 or cfg['grad_accum'] != 1 or cfg['hflip'] != 0:
        parser.error('论文迁移流程要求 batch_size=2、grad_accum=1、hflip=0')
    split_dir = Path(cfg['fewshot_dir'])/f'seed_{args.seed}'
    manifest_path = split_dir/'manifest.json'
    if not manifest_path.exists():
        parser.error('缺少固定小样本划分，请先运行 tools/prepare_fewshot.py')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['seed'] != args.seed:
        parser.error('小样本划分 seed 不一致')
    for field in ('train_data','val_data','test_data'):
        path = Path(cfg[field]['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['source_sha256'][path.name]:
            parser.error(f'划分源数据已变化，请重新生成小样本清单: {path}')
    cfg['seed'] = args.seed
    cfg['train_data'] = {**cfg['train_data'],'path':str(split_dir/f'train_{args.shots}.jsonl')}
    cfg['val_data'] = {**cfg['val_data'],'path':str(split_dir/'val_20.jsonl')}
    cfg['output_dir'] = str(Path(cfg['output_dir'])/f'seed_{args.seed}_shots_{args.shots}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = bool(cfg['amp'] and device == 'cuda')
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    train_samples, val_samples = load_samples(cfg['train_data']), load_samples(cfg['val_data'])
    if len(train_samples) != args.shots or len(val_samples) != 20:
        raise ValueError('小样本清单数量不符合预算')
    for samples in (train_samples,val_samples):
        if len({image_key(s) for s in samples}) != len(samples):
            raise ValueError('每个原始图像只允许一条指代表达')
    if {image_key(s) for s in train_samples} & {image_key(s) for s in val_samples}:
        raise ValueError('小样本训练与验证图像重叠')
    def loader(samples, data_cfg, train):
        dataset = RefSegDataset(samples, data_cfg['image_root'],train=train,
                                 short_side=cfg['short_side'],max_size=cfg['max_size'],hflip=0.)
        return DataLoader(dataset,batch_size=2,shuffle=train,num_workers=cfg['num_workers'],
                           collate_fn=collate_fn)
    training = loader(train_samples,cfg['train_data'],True)
    validation = loader(val_samples,cfg['val_data'],False)
    model, _ = build_model(cfg['gdino_cfg'],None,cfg['bert_path'],device,
                            use_checkpoint=cfg['use_checkpoint'],use_hed=cfg['use_hed'],
                            use_dn=cfg['use_dn'],dn_cfg=cfg['dn'],use_mmscope=cfg['use_mmscope'],
                            mm_cfg=cfg['mm'],use_ref_score=cfg['use_ref_score'])
    source = torch.load(args.source_ckpt,map_location='cpu')
    load_model_state(model,source['model'])
    criterion = build_criterion(cfg)
    optimizer = adaptation_optimizer(model,cfg)
    scaler = build_grad_scaler(amp)
    controller = AdaptationController()
    iterator = iter(training)
    overflow_attempts = 0
    model.train()
    with (output/'adaptation.jsonl').open('w',encoding='utf-8') as log:
        def record(value):
            text = json.dumps(value,ensure_ascii=False)
            log.write(text+'\n')
            log.flush()
            print(text,flush=True)
        record(dict(source=args.source_ckpt,seed=args.seed,shots=args.shots,validation_annotations=20))
        while not controller.stopped:
            try:
                images, targets = next(iterator)
            except StopIteration:
                iterator = iter(training)
                images, targets = next(iterator)
            images = [image.to(device) for image in images]
            captions = [target['caption'] for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
                prediction = model(images,captions,targets=to_device(targets,device))
                loss, _, _ = criterion(prediction,to_device(targets,device))
            if not torch.isfinite(loss):
                raise FloatingPointError('适配训练损失非有限')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_max_norm'])
            if not amp and not torch.isfinite(norm):
                raise FloatingPointError('适配梯度非有限')
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if amp and scaler.get_scale() < before:
                overflow_attempts += 1
                if overflow_attempts >= 20:
                    raise FloatingPointError('连续20次 AMP 更新被跳过')
                continue
            overflow_attempts = 0
            trained_stage = controller.stage
            due = controller.updated()
            if due:
                metrics = evaluate(model,validation,device,amp,pp=cfg['postprocess'],
                                     hed_mask_mode=cfg['hed_mask_mode'])
                event = controller.observe(metrics['mIoU'])
                if event['save_best']:
                    torch.save(dict(model=model.state_dict(),cfg=cfg,metrics=metrics,
                                     source=args.source_ckpt,stage=trained_stage,updates=controller.updates),
                               output/'best.pth')
                if event['transition']:
                    enter_second_stage(optimizer)
                elif event['decay']:
                    for group in optimizer.param_groups:
                        group['lr'] *= .1
                record(dict(updates=controller.updates,trained_stage=trained_stage,
                             next_stage=controller.stage,metrics=metrics,event=event,
                             lr=[g['lr'] for g in optimizer.param_groups],loss=float(loss.detach())))
        if args.eval_test:
            best = torch.load(output/'best.pth',map_location='cpu')
            load_model_state(model,best['model'])
            testing = loader(load_samples(cfg['test_data']),cfg['test_data'],False)
            metrics = evaluate(model,testing,device,amp,pp=cfg['postprocess'],
                                 hed_mask_mode=cfg['hed_mask_mode'])
            record(dict(selected_updates=best['updates'],test=metrics))


if __name__ == '__main__':
    main()
