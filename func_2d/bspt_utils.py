#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
BSPT-MedSAM 2D Utils (修复版)
====================
放置位置: func_2d/bspt_utils.py

修复内容:
  [问题2] PFBE 移除时 BAAM 被连带跳过 → 无 PFBE 时用 fallback 生成 boundary_map
  [问题5] E3_wo_PFBE 配置矛盾 → 现在 BAAM/DSA 不再依赖 PFBE 的存在

使用方法:
---------
在你的 function.py 中:

1. 导入:
   from func_2d.bspt_utils import apply_bspt_to_features, create_bspt_loss

2. 在forward后应用BSPT:
   image_embed = feats[-1]
   if args.use_bspt:
       image_embed, boundary_map, importance = apply_bspt_to_features(image_embed, bspt_modules)

3. 使用BSPT损失:
   if args.use_bspt:
       loss, loss_dict = bspt_loss(pred, masks, epoch=epoch)
   else:
       loss = lossfunc(pred, masks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# 导入BSPT模块
from func_2d.bspt_modules import DAFA, PFBE, BAAM, ULA, DSA, HBALLoss


def _fallback_boundary_map(features: torch.Tensor) -> torch.Tensor:
    """
    ★★★ [修复-问题2] 当 PFBE 不存在时，用简单 Sobel 生成 boundary_map ★★★

    这样 BAAM 和 DSA 在 E3_wo_PFBE 实验中仍能正常工作，
    消融实验才是真正只移除了 PFBE 一个模块。

    Args:
        features: (B, C, H, W) 特征图

    Returns:
        boundary_map: (B, 1, H, W) 边界概率图
    """
    with torch.no_grad():
        x_gray = features.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=features.dtype, device=features.device
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        gx = F.conv2d(x_gray, sobel_x, padding=1)
        gy = F.conv2d(x_gray, sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        boundary_map = torch.sigmoid(5.0 * edge - 2.5)
    return boundary_map


def create_bspt_modules(hidden_dim: int, args) -> Dict[str, nn.Module]:
    """
    创建BSPT模块字典。

    Args:
        hidden_dim: 特征维度 (通常是256)
        args: 配置参数

    Returns:
        modules: 包含所有BSPT模块的字典
    """
    modules = {}

    use_dafa = getattr(args, 'use_dafa', True)
    use_pfbe = getattr(args, 'use_pfbe', True)
    use_baam = getattr(args, 'use_baam', True)
    use_ula = getattr(args, 'use_ula', False)
    use_dsa = getattr(args, 'use_dsa', False)

    dafa_rank = getattr(args, 'dafa_rank', 4)
    ula_compression = getattr(args, 'ula_compression', 16)
    dsa_sparsity = getattr(args, 'dsa_sparsity', 0.25)
    pfbe_scales = getattr(args, 'pfbe_scales', [3, 5])

    if use_dafa:
        modules['dafa'] = DAFA(hidden_dim, rank=dafa_rank)
    if use_pfbe:
        modules['pfbe'] = PFBE(scales=pfbe_scales)
    if use_baam:
        modules['baam'] = BAAM(hidden_dim, num_heads=8)
    if use_ula:
        modules['ula'] = ULA(hidden_dim, compression_ratio=ula_compression)
    if use_dsa:
        modules['dsa'] = DSA(hidden_dim, num_heads=8, sparsity_ratio=dsa_sparsity)

    # 打印信息
    print("\n" + "=" * 50)
    print("BSPT Modules Created")
    print("=" * 50)
    total_params = 0
    for name, module in modules.items():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        print(f"  {name.upper()}: {params:,} params")
    print(f"  Total: {total_params:,} params")
    print("=" * 50 + "\n")

    return modules


def apply_bspt_to_features(
        features: torch.Tensor,
        modules: Dict[str, nn.Module]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    将BSPT模块应用到特征上。

    ★★★ [修复-问题2] 当 PFBE 不存在时，用 fallback 生成 boundary_map ★★★
    这样 BAAM 和 DSA 不会因为 PFBE 缺失而被跳过或降级。

    Args:
        features: 输入特征 (B, C, H, W)
        modules: BSPT模块字典

    Returns:
        enhanced_features: 增强后的特征
        boundary_map: 边界图 (可选)
        importance: 重要性图 (可选)
    """
    boundary_map = None
    importance = None

    # DAFA: 域感知特征适配
    if 'dafa' in modules:
        features = modules['dafa'](features)

    # PFBE: 无参数边界增强
    if 'pfbe' in modules:
        features, boundary_map = modules['pfbe'](features, return_boundary_map=True)
    else:
        # ★★★ [修复-问题2] 即使没有 PFBE，也生成 boundary_map ★★★
        # 这样后续的 BAAM 和 DSA 仍能正常使用 boundary_map
        # 使用 no_grad 的简单 Sobel，不影响 PFBE 消融的公平性
        if 'baam' in modules or 'dsa' in modules:
            boundary_map = _fallback_boundary_map(features)

    # BAAM: 边界感知注意力
    # ★★★ [修复-问题2] 现在 boundary_map 始终可用 (来自 PFBE 或 fallback) ★★★
    if 'baam' in modules and boundary_map is not None:
        features = modules['baam'](features, boundary_map)

    # ULA: 超轻量适配器
    if 'ula' in modules:
        features = modules['ula'](features)

    # DSA: 动态稀疏注意力
    # ★★★ [修复-问题2] boundary_map 始终可用，DSA 不再降级运行 ★★★
    if 'dsa' in modules:
        features, importance = modules['dsa'](features, boundary_map)

    return features, boundary_map, importance


def create_bspt_loss(args, device) -> HBALLoss:
    """
    创建BSPT损失函数。

    注意: HBALLoss 默认参数已与 cfg_bspt.py 统一 (lambda_bce=0.5, lambda_boundary=0.3, lambda_hd=0.1)
    """
    return HBALLoss(
        lambda_dice=getattr(args, 'lambda_dice', 1.0),
        lambda_bce=getattr(args, 'lambda_bce', 0.5),
        lambda_boundary=getattr(args, 'lambda_boundary', 0.3),
        lambda_hd=getattr(args, 'lambda_hd', 0.1),
        warmup_epochs=getattr(args, 'hbal_warmup', 10)
    ).to(device)


def get_bspt_parameters(modules: Dict[str, nn.Module]) -> list:
    """
    获取所有BSPT模块的可训练参数。
    """
    params = []
    for module in modules.values():
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params