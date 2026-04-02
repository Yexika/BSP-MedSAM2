#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
BSPT-MedSAM 2D Modules (论文对齐+修复合并版)
=============================================
放置位置: func_2d/bspt_modules.py

合并策略: 论文对齐改进 + 稳定性修复
  [DAFA]  论文对齐: 加 GELU 激活
  [PFBE]  论文对齐: 加归一化, scales=[3,5,7]
  [BAAM]  合并: F*(1+B) (稳定性) + Conv3×3 (论文对齐)
  [ULA]   论文对齐: 加 GELU, 加可学习α
  [DSA]   论文对齐: 加性融合 S + λ·B
  [初始化] 修复: BAAM/ULA/DSA 小随机初始化
  [PFBE]  修复: fallback 保证消融公平性
  [DSA]   修复: LayerNorm 仅 top-K

包含所有2D BSPT模块:
- DAFA: Domain-Aware Feature Adapter
- PFBE: Parameter-Free Boundary Enhancement
- BAAM: Boundary-Aware Attention Module
- ULA: Ultra-Lightweight Adapter
- DSA: Dynamic Sparse Attention
- HBALLoss: Hybrid Boundary-Aware Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# =============================================================================
# DAFA: Domain-Aware Feature Adapter
# =============================================================================

class DAFA(nn.Module):
    """
    Domain-Aware Feature Adapter using LoRA-style low-rank decomposition.

    论文公式:
        F_down = F^token · W_down
        F_adapt = σ(F_down) · W_up      ← σ = GELU
        F_out = F^token + α · F_adapt

    [论文对齐] 加 GELU 激活函数
    """

    def __init__(self, in_features: int, rank: int = 4, alpha_init: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.rank = rank

        self.W_down = nn.Linear(in_features, rank, bias=False)
        self.activation = nn.GELU()  # ★ [论文对齐]
        self.W_up = nn.Linear(rank, in_features, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        nn.init.kaiming_uniform_(self.W_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_spatial = len(x.shape) == 4

        if is_spatial:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            delta = self.W_up(self.activation(self.W_down(x_flat)))
            delta = delta.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            delta = self.W_up(self.activation(self.W_down(x)))

        return x + self.alpha * delta


# =============================================================================
# PFBE: Parameter-Free Boundary Enhancement
# =============================================================================

class PFBE(nn.Module):
    """
    Parameter-Free Boundary Enhancement using multi-scale Sobel and Laplacian.

    论文公式:
        B = (1/|S|) Σ E^(s) + |F_avg * L|      (S={3,5,7})
        A_boundary = σ(γ · (B - μ_B) / σ_B + β)
        F_enhanced = F ⊙ (1 + A_boundary)

    [论文对齐] 加归一化 (B - μ) / σ
    [论文对齐] scales 默认 [3,5,7]
    """

    def __init__(self, scales: list = None, gamma: float = 5.0, beta: float = -2.5):
        super().__init__()
        self.scales = scales or [3, 5, 7]  # ★ [论文对齐]
        self.gamma = gamma
        self.beta = beta

        for scale in self.scales:
            sobel_x, sobel_y = self._create_sobel_kernels(scale)
            self.register_buffer(f'sobel_x_{scale}', sobel_x)
            self.register_buffer(f'sobel_y_{scale}', sobel_y)

        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))

    def _create_sobel_kernels(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if size == 3:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        elif size == 5:
            sobel_x = torch.tensor([
                [-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6],
                [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]
            ], dtype=torch.float32)
        elif size == 7:
            sobel_x = torch.tensor([
                [-1, -4, -5, 0, 5, 4, 1], [-6, -24, -30, 0, 30, 24, 6],
                [-15, -60, -75, 0, 75, 60, 15], [-20, -80, -100, 0, 100, 80, 20],
                [-15, -60, -75, 0, 75, 60, 15], [-6, -24, -30, 0, 30, 24, 6],
                [-1, -4, -5, 0, 5, 4, 1]
            ], dtype=torch.float32)
        else:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)

        sobel_y = sobel_x.t()
        return sobel_x.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0)

    def _compute_boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        edge_maps = []
        for scale in self.scales:
            sobel_x = getattr(self, f'sobel_x_{scale}')
            sobel_y = getattr(self, f'sobel_y_{scale}')
            padding = scale // 2

            grad_x = F.conv2d(x_gray, sobel_x, padding=padding)
            grad_y = F.conv2d(x_gray, sobel_y, padding=padding)
            edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edge_maps.append(edge)

        laplacian_edge = torch.abs(F.conv2d(x_gray, self.laplacian, padding=1))
        edge_maps.append(laplacian_edge)

        combined = torch.cat(edge_maps, dim=1).mean(dim=1, keepdim=True)

        # ★ [论文对齐] A = σ(γ · (B - μ_B) / σ_B + β)
        mu = combined.mean(dim=(2, 3), keepdim=True)
        sigma = combined.std(dim=(2, 3), keepdim=True) + 1e-6
        normalized = (combined - mu) / sigma
        boundary_map = torch.sigmoid(self.gamma * normalized + self.beta)

        return boundary_map

    def forward(self, x: torch.Tensor, return_boundary_map: bool = False):
        boundary_map = self._compute_boundary_map(x)
        attention = 1.0 + boundary_map
        out = x * attention

        if return_boundary_map:
            return out, boundary_map
        return out


# =============================================================================
# BAAM: Boundary-Aware Attention Module
# =============================================================================

class BAAM(nn.Module):
    """
    Boundary-Aware Attention Module with cross-attention.

    [论文对齐] 加 Conv3×3 depthwise 边界/区域特征提取
    [稳定性修复] 用 F*(1+B) 代替 F*B, 避免近零 token
    [初始化修复] out_proj xavier(gain=0.1), conv normal(std=0.01)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ★ [论文对齐] Conv3×3 depthwise
        self.boundary_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.region_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.q_boundary = nn.Linear(dim, dim)
        self.k_region = nn.Linear(dim, dim)
        self.v_region = nn.Linear(dim, dim)
        self.q_region = nn.Linear(dim, dim)
        self.k_boundary = nn.Linear(dim, dim)
        self.v_boundary = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.lambda_b = nn.Parameter(torch.tensor(0.5))
        self.lambda_r = nn.Parameter(torch.tensor(0.5))

        # ★ [初始化修复]
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.boundary_conv.weight, std=0.01)
        nn.init.zeros_(self.boundary_conv.bias)
        nn.init.normal_(self.region_conv.weight, std=0.01)
        nn.init.zeros_(self.region_conv.bias)

    def forward(self, x: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        boundary_weight = boundary_map.expand(-1, C, -1, -1)

        # ★★★ [合并] F*(1+B) 数值安全 + Conv3×3 论文对齐 ★★★
        fb_weighted = x * (1.0 + boundary_weight)                    # F ⊙ (1+B)
        F_boundary = fb_weighted + self.boundary_conv(fb_weighted)   # + Conv(F⊙(1+B))

        fr_weighted = x * (2.0 - boundary_weight)                   # F ⊙ (2-B), 互补
        F_region = fr_weighted + self.region_conv(fr_weighted)       # + Conv(F⊙(2-B))

        F_b_flat = F_boundary.permute(0, 2, 3, 1).reshape(B, H * W, C)
        F_r_flat = F_region.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Boundary -> Region cross-attention
        q_b = self.q_boundary(F_b_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_r = self.k_region(F_r_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_r = self.v_region(F_r_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_b2r = F.softmax((q_b @ k_r.transpose(-2, -1)) * self.scale, dim=-1)
        out_b2r = (self.dropout(attn_b2r) @ v_r).permute(0, 2, 1, 3).reshape(B, H * W, C)

        # Region -> Boundary cross-attention
        q_r = self.q_region(F_r_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_b = self.k_boundary(F_b_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_b = self.v_boundary(F_b_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_r2b = F.softmax((q_r @ k_b.transpose(-2, -1)) * self.scale, dim=-1)
        out_r2b = (self.dropout(attn_r2b) @ v_b).permute(0, 2, 1, 3).reshape(B, H * W, C)

        out = self.out_proj(self.lambda_b * out_b2r + self.lambda_r * out_r2b)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out = self.norm(out + x_flat)

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


# =============================================================================
# ULA: Ultra-Lightweight Adapter
# =============================================================================

class ULA(nn.Module):
    """
    Ultra-Lightweight Adapter: depthwise conv + SE.

    [论文对齐] DWConv 后加 GELU, SE 中 ReLU→GELU, 加可学习 α
    [初始化修复] dw_conv normal(std=0.01)
    """

    def __init__(self, dim: int, compression_ratio: int = 16, kernel_size: int = 3,
                 alpha_init: float = 0.1):
        super().__init__()
        reduced_dim = max(dim // compression_ratio, 8)

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.spatial_act = nn.GELU()  # ★ [论文对齐]

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, reduced_dim), nn.GELU(),  # ★ [论文对齐] ReLU → GELU
            nn.Linear(reduced_dim, dim), nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # ★ [论文对齐]
        self.norm = nn.LayerNorm(dim)

        # ★ [初始化修复]
        nn.init.normal_(self.dw_conv.weight, std=0.01)
        nn.init.zeros_(self.dw_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        spatial = self.spatial_act(self.dw_conv(x))
        channel = self.se(x).view(B, C, 1, 1)
        out = x + self.alpha * (spatial * channel)
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


# =============================================================================
# DSA: Dynamic Sparse Attention
# =============================================================================

class DSA(nn.Module):
    """
    Dynamic Sparse Attention.

    [论文对齐] importance 融合: 加性 S + λ·B
    [初始化修复] out_proj xavier(gain=0.1)
    [稳定性修复] LayerNorm 仅对 top-K token
    """

    def __init__(self, dim: int, num_heads: int = 8, sparsity_ratio: float = 0.25,
                 boundary_weight: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_ratio = sparsity_ratio
        self.boundary_weight = boundary_weight

        self.importance_net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid()
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # ★ [初始化修复]
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, boundary_map: torch.Tensor = None):
        B, C, H, W = x.shape
        N, K = H * W, max(int(H * W * self.sparsity_ratio), 1)

        importance = self.importance_net(x)
        if boundary_map is not None:
            # ★ [论文对齐] S_final = S + λ·B (加性融合)
            importance = importance + self.boundary_weight * boundary_map

        _, top_idx = torch.topk(importance.view(B, N), K, dim=1)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
        x_sparse = x_flat[batch_idx, top_idx]

        q = self.q_proj(x_sparse).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_sparse).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_sparse).reshape(B, K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out_sparse = self.out_proj((attn @ v).permute(0, 2, 1, 3).reshape(B, K, C))

        # ★ [稳定性修复] 仅对 top-K token 做残差+norm
        enhanced_sparse = self.norm(out_sparse + x_sparse)
        out_full = x_flat.clone()
        out_full[batch_idx, top_idx] = enhanced_sparse

        out = out_full.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out, importance


# =============================================================================
# HBAL: Hybrid Boundary-Aware Loss
# =============================================================================

class HBALLoss(nn.Module):
    """
    Hybrid Boundary-Aware Loss (HBAL)
    L = λ₁L_dice + λ₂L_bce + λ₃L_boundary + λ₄L_hd
    """

    def __init__(self, lambda_dice=1.0, lambda_bce=0.5, lambda_boundary=0.3,
                 lambda_hd=0.1, warmup_epochs=10):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_boundary = lambda_boundary
        self.lambda_hd = lambda_hd
        self.warmup_epochs = warmup_epochs

    def _dice_loss(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def _bce_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def _boundary_loss(self, pred, target, kernel_size=3):
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device)
        padding = kernel_size // 2

        dilated = (F.conv2d(target, kernel, padding=padding) > 0).float()
        eroded = (F.conv2d(target, kernel, padding=padding) >= kernel_size ** 2).float()
        boundary = (dilated - eroded).clamp(0, 1)

        boundary_weight = boundary + 1e-6

        return F.binary_cross_entropy_with_logits(
            pred * boundary_weight,
            target * boundary_weight,
            reduction='mean'
        )

    def _hausdorff_loss(self, pred, target):
        pred_sig = torch.sigmoid(pred)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        pred_grad_x = F.conv2d(pred_sig, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_sig, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)

        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)

        return F.mse_loss(pred_edge, target_edge)

    def forward(self, pred, target, epoch=0):
        warmup_factor = min(1.0, epoch / max(1, self.warmup_epochs))

        dice_loss = self._dice_loss(pred, target)
        bce_loss = self._bce_loss(pred, target)
        boundary_loss = self._boundary_loss(pred, target)
        hd_loss = self._hausdorff_loss(pred, target)

        total_loss = (
                self.lambda_dice * dice_loss +
                self.lambda_bce * bce_loss +
                self.lambda_boundary * warmup_factor * boundary_loss +
                self.lambda_hd * warmup_factor * hd_loss
        )

        loss_dict = {
            'dice': dice_loss.item(),
            'bce': bce_loss.item(),
            'boundary': boundary_loss.item(),
            'hd': hd_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


class ImportanceLoss(nn.Module):
    """Auxiliary loss for DSA importance learning."""

    def __init__(self):
        super().__init__()

    def forward(self, importance, mask):
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        dilated = (F.conv2d(mask, kernel, padding=1) > 0).float()
        eroded = (F.conv2d(mask, kernel, padding=1) >= 9).float()
        boundary = (dilated - eroded).clamp(0, 1)
        return F.binary_cross_entropy_with_logits(importance, boundary)


# =============================================================================
# Boundary Map Fallback (用于消融实验)
# =============================================================================

def compute_boundary_map_fallback(features: torch.Tensor) -> torch.Tensor:
    """
    ★★★ [修复] 当 PFBE 被消融时，为 BAAM/DSA 生成 boundary_map ★★★

    用简单 Sobel 3×3 + 归一化，不增强特征，不引入参数。
    no_grad 确保不参与反向传播，不影响消融公平性。
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

        mu = edge.mean(dim=(2, 3), keepdim=True)
        sigma = edge.std(dim=(2, 3), keepdim=True) + 1e-6
        boundary_map = torch.sigmoid(5.0 * (edge - mu) / sigma - 2.5)

    return boundary_map


# =============================================================================
# BSPT-MedSAM 2D Wrapper
# =============================================================================

class BSPTMedSAM(nn.Module):
    """BSPT-MedSAM 2D: Integrating BSPT modules with MedSAM2."""

    def __init__(self, base_model, args, freeze_encoder=True,
                 use_dafa=True, use_pfbe=True, use_baam=True, use_ula=True, use_dsa=True,
                 dafa_rank=4, ula_compression=16, dsa_sparsity=0.25, pfbe_scales=None):
        super().__init__()

        self.base_model = base_model
        self.args = args
        self.use_dafa = use_dafa
        self.use_pfbe = use_pfbe
        self.use_baam = use_baam
        self.use_ula = use_ula
        self.use_dsa = use_dsa
        self.hidden_dim = base_model.hidden_dim

        if freeze_encoder:
            for n, p in base_model.named_parameters():
                if 'image_encoder' in n:
                    p.requires_grad = False
            print("[BSPT-MedSAM] Image encoder frozen")

        if pfbe_scales is None:
            pfbe_scales = [3, 5, 7]

        if use_dafa: self.dafa = DAFA(self.hidden_dim, rank=dafa_rank)
        if use_pfbe: self.pfbe = PFBE(scales=pfbe_scales)
        if use_baam: self.baam = BAAM(self.hidden_dim, num_heads=8)
        if use_ula: self.ula = ULA(self.hidden_dim, compression_ratio=ula_compression)
        if use_dsa: self.dsa = DSA(self.hidden_dim, num_heads=8, sparsity_ratio=dsa_sparsity)

        self._print_info()

    def _print_info(self):
        print("\n" + "=" * 50)
        print("BSPT-MedSAM 2D Configuration (Paper-Aligned + Fixed)")
        print("=" * 50)
        total = 0
        for name in ['dafa', 'pfbe', 'baam', 'ula', 'dsa']:
            if hasattr(self, name):
                p = sum(x.numel() for x in getattr(self, name).parameters() if x.requires_grad)
                total += p
                print(f"  {name.upper()}: {p:,} params")
        print(f"  Total BSPT: {total:,}")
        print("=" * 50 + "\n")

    def apply_bspt_modules(self, features):
        """
        ★★★ [修复] PFBE 消融时仍为下游模块提供 boundary_map ★★★
        """
        boundary_map = None
        importance = None

        # DAFA
        if self.use_dafa and hasattr(self, 'dafa'):
            features = self.dafa(features)

        # PFBE
        if self.use_pfbe and hasattr(self, 'pfbe'):
            features, boundary_map = self.pfbe(features, return_boundary_map=True)
        else:
            # ★ fallback: 不增强特征, 仅生成 boundary_map
            boundary_map = compute_boundary_map_fallback(features)

        # BAAM
        if self.use_baam and hasattr(self, 'baam') and boundary_map is not None:
            features = self.baam(features, boundary_map)

        # ULA
        if self.use_ula and hasattr(self, 'ula'):
            features = self.ula(features)

        # DSA
        if self.use_dsa and hasattr(self, 'dsa'):
            features, importance = self.dsa(features, boundary_map)

        return features, boundary_map, importance

    def forward_image(self, images):
        return self.base_model.forward_image(images)

    def _prepare_backbone_features(self, x):
        return self.base_model._prepare_backbone_features(x)

    def _encode_new_memory(self, *a, **k):
        return self.base_model._encode_new_memory(*a, **k)

    def memory_attention(self, *a, **k):
        return self.base_model.memory_attention(*a, **k)

    @property
    def sam_prompt_encoder(self):
        return self.base_model.sam_prompt_encoder

    @property
    def sam_mask_decoder(self):
        return self.base_model.sam_mask_decoder


if __name__ == '__main__':
    print("Testing BSPT 2D modules (Paper-Aligned + Fixed)...")
    x = torch.randn(2, 256, 16, 16)

    for name, mod in [('DAFA', DAFA(256)), ('PFBE', PFBE()), ('BAAM', BAAM(256)),
                      ('ULA', ULA(256)), ('DSA', DSA(256))]:
        if name == 'PFBE':
            out, bmap = mod(x, return_boundary_map=True)
            print(f"  boundary_map: mean={bmap.mean():.4f}, max={bmap.max():.4f}, "
                  f"min={bmap.min():.4f}, std={bmap.std():.4f}")
        elif name == 'BAAM':
            out = mod(x, bmap)
        elif name == 'DSA':
            out, _ = mod(x, bmap)
        else:
            out = mod(x)
        print(f"{name}: {x.shape} -> {out.shape}")

    # 测试 fallback
    bmap_fb = compute_boundary_map_fallback(x)
    print(f"\nFallback boundary_map: shape={bmap_fb.shape}, "
          f"mean={bmap_fb.mean():.4f}, max={bmap_fb.max():.4f}")

    # 验证参数量
    print("\n--- Parameter Count ---")
    for name, mod in [('DAFA', DAFA(256)), ('PFBE', PFBE()), ('BAAM', BAAM(256)),
                      ('ULA', ULA(256)), ('DSA', DSA(256))]:
        p = sum(x.numel() for x in mod.parameters() if x.requires_grad)
        print(f"  {name}: {p:,} params")

    print("\nAll paper-aligned + fixed module tests passed!")