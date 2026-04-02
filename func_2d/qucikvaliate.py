#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
quick_validate.py - 快速验证3个新数据集的训练流程 (~5分钟)
========================================================
Kvasir-SEG和DSB18已有权重，只验证CVC-ClinicDB/ISIC17/ISIC18。

检查内容:
  1. 数据集加载正常（mask != image）
  2. BSPT 3模块版正常（无ULA/DSA）
  3. 损失正常、梯度正常
  4. 权重保存/加载正常

用法:
  cd /root/autodl-tmp/BSPT-Medsam
  python quick_validate.py
"""

import os, sys, time, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

BASE_PATH = '/root/autodl-tmp/BSPT-Medsam'
sys.path.insert(0, BASE_PATH)

import cfg_bspt as cfg
from func_2d.utils import get_network
from func_2d.dataset_modified import MultiDataset
from func_2d.bspt_utils import create_bspt_modules, create_bspt_loss, apply_bspt_to_features

DATA_ROOT = '/root/autodl-tmp/datasets'

# 只验证需要训练的3个
DATASETS = ['CVC-ClinicDB', 'ISIC17', 'ISIC18']


def validate(args, ds_name, device):
    data_path = os.path.join(DATA_ROOT, ds_name)
    warns = []

    # 1. 文件
    if not os.path.exists(data_path):
        return False, [f"路径不存在: {data_path}"], warns
    print(f"  [1/5] 目录: {sorted(os.listdir(data_path))}")

    # 2. 数据集加载
    print(f"  [2/5] 加载数据...")
    try:
        train_ds = MultiDataset(args, data_path, mode='Training', prompt='click', seed=args.seed)
        test_ds = MultiDataset(args, data_path, mode='Test', prompt='click', seed=args.seed)
        loader = DataLoader(train_ds, batch_size=min(args.b, 2), shuffle=True,
                             num_workers=0, drop_last=True)
        print(f"       ✅ 训练:{len(train_ds)} 测试:{len(test_ds)}")
        if len(train_ds) == 0:
            return False, ["训练集为空"], warns
    except Exception as e:
        return False, [f"加载失败: {e}"], warns

    # 3. mask验证
    print(f"  [3/5] 验证mask...")
    try:
        batch = next(iter(loader))
        imgs, masks = batch['image'], batch['mask']
        # ★ 防止mask=image的致命bug
        if imgs.shape == masks.shape:
            sim = F.cosine_similarity(imgs.float().reshape(1,-1), masks.float().reshape(1,-1)).item()
            if sim > 0.95:
                return False, [f"mask≈image (cos={sim:.4f})! mask路径错误!"], warns
        print(f"       ✅ img:{imgs.shape} mask:{masks.shape} "
              f"mask范围[{masks.min():.2f},{masks.max():.2f}]")
    except Exception as e:
        return False, [f"数据验证失败: {e}"], warns

    # 4. BSPT + 2epoch训练
    print(f"  [4/5] 训练2epoch×3batch...")
    try:
        net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
        bspt = create_bspt_modules(net.hidden_dim, args)
        for n, m in bspt.items():
            if isinstance(m, nn.Module): bspt[n] = m.to(device)
        loss_fn = create_bspt_loss(args, device)

        active = [n for n, m in bspt.items() if isinstance(m, nn.Module)]
        print(f"       模块: {active}")
        if any(n in ('ula','dsa') for n in active):
            warns.append("ULA/DSA仍被创建!")

        net.train()
        for m in bspt.values():
            if isinstance(m, nn.Module): m.train()

        all_p = list(filter(lambda p: p.requires_grad, net.parameters()))
        for m in bspt.values():
            if isinstance(m, nn.Module):
                all_p.extend(filter(lambda p: p.requires_grad, m.parameters()))
        opt = torch.optim.AdamW(all_p, lr=1e-4, weight_decay=0.01)

        feat_sizes = [(args.image_size//4,)*2, (args.image_size//8,)*2, (args.image_size//16,)*2]

        for epoch in range(2):
            total_loss, nb = 0, 0
            for bi, pack in enumerate(loader):
                if bi >= 3: break
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    im = pack['image'].to(device, dtype=torch.float32)
                    mk = pack['mask'].to(device, dtype=torch.float32)
                    if 'pt' in pack:
                        pt = pack['pt'].to(device).unsqueeze(1)
                        pl = pack['p_label'].to(device).unsqueeze(1)
                    else:
                        pt = torch.tensor([[[512.,512.]]], device=device).expand(im.shape[0],1,2)
                        pl = torch.ones(im.shape[0],1, device=device, dtype=torch.int)

                    bo = net.forward_image(im)
                    _, vf, vpe, _ = net._prepare_backbone_features(bo)
                    B = vf[-1].size(1)
                    vf[-1] += torch.zeros(1,B,net.hidden_dim, device=device)
                    vpe[-1] += torch.zeros(1,B,net.hidden_dim, device=device)
                    feats = [f.permute(1,2,0).view(B,-1,*fs) for f,fs in zip(vf[::-1],feat_sizes[::-1])][::-1]
                    ie = feats[-1]
                    ie, _, _ = apply_bspt_to_features(ie, bspt)
                    hrf = feats[:-1]
                    se, de = net.sam_prompt_encoder(points=(pt.float(),pl.int()), boxes=None, masks=None, batch_size=B)
                    lrm, _, _, _ = net.sam_mask_decoder(
                        image_embeddings=ie, image_pe=net.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                        multimask_output=False, repeat_image=False, high_res_features=hrf)
                    pred = F.interpolate(lrm, size=(args.out_size, args.out_size))
                    gt = F.interpolate(mk.float(), size=(args.out_size, args.out_size))
                    loss, ld = loss_fn(pred, gt, epoch=epoch)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_p, 1.0)
                opt.step()
                total_loss += loss.item(); nb += 1

            print(f"       Epoch{epoch}: loss={total_loss/max(nb,1):.4f} "
                  f"(d={ld['dice']:.3f} b={ld['bce']:.3f} bd={ld['boundary']:.3f} h={ld['hd']:.3f})")

        if np.isnan(total_loss):
            return False, ["Loss=NaN!"], warns
        print(f"       ✅ 训练正常")

    except Exception as e:
        import traceback; traceback.print_exc()
        return False, [f"训练失败: {e}"], warns

    # 5. 保存/加载
    print(f"  [5/5] 保存/加载...")
    try:
        tmp = f'/tmp/qv_{ds_name}.pth'
        torch.save({
            'model': net.state_dict(),
            'bspt_modules': {n: m.state_dict() for n,m in bspt.items() if isinstance(m,nn.Module)},
        }, tmp)
        ck = torch.load(tmp, map_location=device)
        assert 'model' in ck and 'bspt_modules' in ck
        os.remove(tmp)
        print(f"       ✅ 保存/加载正常")
    except Exception as e:
        return False, [f"保存失败: {e}"], warns

    del net, bspt, opt
    torch.cuda.empty_cache()
    return True, [], warns


def main():
    args = cfg.parse_args()
    for k, v in {
        'use_bspt':True, 'use_dafa':True, 'use_pfbe':True,
        'use_baam':True, 'use_ula':False, 'use_dsa':False,
        'dafa_rank':4, 'pfbe_scales':[3,5],
        'lambda_dice':1.0, 'lambda_bce':0.5,
        'lambda_boundary':0.3, 'lambda_hd':0.1, 'hbal_warmup':10,
    }.items():
        setattr(args, k, v)

    device = torch.device('cuda', args.gpu_device)

    print("=" * 60)
    print("  快速验证 (3模块: DAFA+PFBE+BAAM)")
    print(f"  数据集: {DATASETS}")
    print("=" * 60)

    results = []
    t0 = time.time()

    for ds in DATASETS:
        print(f"\n{'━'*50}")
        print(f"  {ds}")
        print(f"{'━'*50}")
        start = time.time()
        ok, errs, warns = validate(args, ds, device)
        results.append({'ds':ds, 'ok':ok, 'err':errs, 'warn':warns, 't':time.time()-start})

    print(f"\n\n{'='*60}")
    print("  结果汇总")
    print(f"{'='*60}")
    all_ok = True
    for r in results:
        print(f"  {'✅' if r['ok'] else '❌'} {r['ds']:<15} ({r['t']:.0f}s)")
        for w in r['warn']: print(f"      ⚠️  {w}")
        for e in r['err']:  print(f"      ❌ {e}")
        if not r['ok']: all_ok = False

    print(f"\n  总耗时: {time.time()-t0:.0f}s")
    if all_ok:
        print("\n  🎉 全部通过!")
        print("  nohup python run_all_auto.py > auto_train.log 2>&1 &")
    else:
        print("\n  ⚠️  有失败，先修复!")

    return all_ok

if __name__ == '__main__':
    sys.exit(0 if main() else 1)