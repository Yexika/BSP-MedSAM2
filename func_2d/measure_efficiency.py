#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
measure_efficiency.py - 模型效率测量 (表8 & 表9) [修复版]
=========================================================
修复: FLOPs用fvcore (Meta官方库，SAM2自带依赖) 替代thop
     thop对Hiera/cross-attention支持差，返回错误值

用法:
  cd /root/autodl-tmp/BSPT-Medsam
  python measure_efficiency.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BSPT_PATH = '/root/autodl-tmp/BSPT-Medsam'
sys.path.insert(0, BSPT_PATH)

IMAGE_SIZE = 1024


# ============================================================
# 1. 参数量统计
# ============================================================

def count_params_by_component(model):
    components = {
        'image_encoder': {'total': 0, 'trainable': 0},
        'memory_attention': {'total': 0, 'trainable': 0},
        'memory_encoder': {'total': 0, 'trainable': 0},
        'prompt_encoder': {'total': 0, 'trainable': 0},
        'mask_decoder': {'total': 0, 'trainable': 0},
        'other': {'total': 0, 'trainable': 0},
    }
    for name, param in model.named_parameters():
        num = param.numel()
        matched = False
        for comp in ['image_encoder', 'memory_attention', 'memory_encoder',
                     'sam_prompt_encoder', 'sam_mask_decoder']:
            if comp in name:
                key = comp.replace('sam_', '')
                if key not in components:
                    key = 'other'
                components[key]['total'] += num
                if param.requires_grad:
                    components[key]['trainable'] += num
                matched = True
                break
        if not matched:
            components['other']['total'] += num
            if param.requires_grad:
                components['other']['trainable'] += num
    return components


def count_bspt_detail(bspt_modules):
    details = {}
    total = 0
    for name, module in bspt_modules.items():
        if isinstance(module, nn.Module):
            p = sum(x.numel() for x in module.parameters())
            details[name.upper()] = p
            total += p
        else:
            details[name.upper()] = 0
    details['_total'] = total
    return details


# ============================================================
# 2. FLOPs计算 (三种方案，优先级递减)
# ============================================================

def measure_flops_fvcore(model, input_shape=(1, 3, 1024, 1024), device='cuda'):
    """方案A: 用fvcore (SAM2项目自带依赖，支持最好)"""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        dummy = torch.randn(*input_shape, device=device)
        flops_analyzer = FlopCountAnalysis(model, dummy)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        total_flops = flops_analyzer.total()
        return total_flops, 'fvcore'
    except ImportError:
        return None, 'fvcore not installed'
    except Exception as e:
        return None, f'fvcore error: {e}'


def measure_flops_manual_encoder(model_name='hiera_small'):
    """方案B: 手动计算Hiera encoder FLOPs (理论值)

    SAM2用的是Hiera-S (sam2.1_hiera_small):
      - Stages: [1, 2, 11, 2] blocks
      - Dims: [96, 192, 384, 768]
      - Heads: [1, 2, 4, 8]
      - Window sizes变化

    每个block的FLOPs ≈
      Attention: 4*N*D² (QKV+proj) + 2*N²*D (attention matmul)
      MLP: 8*N*D² (两层MLP, expansion=4)
    其中 N=token数, D=dim
    """
    # Hiera-Small 配置 (从SAM2代码)
    stages = [1, 2, 11, 2]
    dims = [96, 192, 384, 768]

    # 输入 1024x1024, patch_size=16 in Hiera → 初始 64x64 = 4096 tokens
    # Stage间有2x2 pooling → tokens: 4096, 1024, 256, 64
    tokens_per_stage = [4096, 1024, 256, 64]

    total_flops = 0

    for stage_idx, (n_blocks, dim, n_tokens) in enumerate(
            zip(stages, dims, tokens_per_stage)):
        for _ in range(n_blocks):
            # Attention: QKV projection + output projection
            attn_proj = 4 * n_tokens * dim * dim * 2  # ×2 for multiply-add
            # Attention matmul (windowed, but upper bound with global)
            attn_matmul = 2 * n_tokens * n_tokens * dim * 2  # 实际是windowed，这里估上界
            # 对于windowed attention，实际 ≈ 2 * n_tokens * window_size² * dim
            # Hiera用的window_size=8或14，这里用合理估计
            window_area = min(n_tokens, 8 * 8)  # 窗口大小估计
            attn_matmul_windowed = 2 * n_tokens * window_area * dim * 2

            # MLP: dim → 4*dim → dim
            mlp_flops = 2 * n_tokens * dim * (4 * dim) * 2  # 两层

            block_flops = attn_proj + attn_matmul_windowed + mlp_flops
            total_flops += block_flops

        # Pooling between stages (relatively small)
        if stage_idx < len(stages) - 1:
            total_flops += n_tokens * dim * dims[stage_idx + 1] * 2  # conv projection

    # Patch embedding: 1024x1024x3 → 64x64x96
    patch_embed_flops = (1024 * 1024 * 3 * 96 * 16 * 16) // (16 * 16)  # 简化
    total_flops += patch_embed_flops * 2

    return total_flops


def measure_flops_baam(dim=256, n_tokens=4096):
    """手动计算BAAM的FLOPs

    BAAM = 双向 cross-attention:
      Boundary→Region: Q(boundary) × K(region) × V(region)
      Region→Boundary: Q(region) × K(boundary) × V(boundary)

    每个方向:
      QKV投影: 3 × N × D × D × 2 (multiply-add)
      Attention: N × N × D × 2 (QK^T) + N × N × D × 2 (softmax×V)
      Output projection: N × D × D × 2
    """
    # 每个方向
    qkv_proj = 3 * n_tokens * dim * dim * 2
    attn_matmul = 2 * n_tokens * n_tokens * dim * 2  # QK^T + softmax*V
    out_proj = n_tokens * dim * dim * 2

    per_direction = qkv_proj + attn_matmul + out_proj

    # 双向
    total = 2 * per_direction

    # LayerNorm + residual (小)
    total += n_tokens * dim * 4  # approx

    return total


def measure_flops_dafa(dim=256, rank=4, n_tokens=4096):
    """DAFA FLOPs: 两个低秩矩阵"""
    # W_down: N × D × r × 2
    # W_up: N × r × D × 2
    return n_tokens * (dim * rank + rank * dim) * 2


def measure_flops_pfbe(n_tokens=4096, channels=256):
    """PFBE FLOPs: Sobel卷积 (非常小)"""
    # 3x3 Sobel: H×W × 9 × 2
    # 5x5 Sobel: H×W × 25 × 2
    # Laplacian: H×W × 9 × 2
    h = w = int(n_tokens ** 0.5)  # 64
    return h * w * (9 + 9 + 25 + 25 + 9) * 2


# ============================================================
# 3. 推理时间测量
# ============================================================

def measure_inference_time(model, bspt_modules=None, num_warmup=10, num_runs=50, device='cuda'):
    model.eval()
    if bspt_modules:
        for m in bspt_modules.values():
            if isinstance(m, nn.Module):
                m.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, dtype=torch.float32)
    pt = torch.tensor([[[512.0, 512.0]]], device=device)
    pt_label = torch.ones(1, 1, device=device, dtype=torch.int)
    feat_sizes = [(IMAGE_SIZE // 4,) * 2, (IMAGE_SIZE // 8,) * 2, (IMAGE_SIZE // 16,) * 2]

    def run_forward():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(dummy)
            _, vision_feats, vision_pos_embeds, _ = model._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)
            vision_feats[-1] = vision_feats[-1] + torch.zeros(1, B, model.hidden_dim, device=device)
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.zeros(1, B, model.hidden_dim, device=device)
            feats = [feat.permute(1, 2, 0).view(B, -1, *fs)
                     for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            if bspt_modules is not None:
                from func_2d.bspt_utils import apply_bspt_to_features
                image_embed, _, _ = apply_bspt_to_features(image_embed, bspt_modules)
            high_res_feats = feats[:-1]
            se, de = model.sam_prompt_encoder(points=(pt, pt_label), boxes=None, masks=None, batch_size=B)
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False,
                high_res_features=high_res_feats)
            F.interpolate(low_res_masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    for _ in range(num_warmup):
        run_forward()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_forward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    t = np.array(times)
    return {'mean': float(np.mean(t)), 'std': float(np.std(t)), 'median': float(np.median(t))}


# ============================================================
# 4. 分段计时 (定位BAAM开销)
# ============================================================

def measure_segmented_time(model, bspt_modules=None, num_runs=30, device='cuda'):
    """分段测时间: encoder / BSPT / decoder 各花多少"""
    model.eval()
    if bspt_modules:
        for m in bspt_modules.values():
            if isinstance(m, nn.Module): m.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, dtype=torch.float32)
    pt = torch.tensor([[[512.0, 512.0]]], device=device)
    pt_label = torch.ones(1, 1, device=device, dtype=torch.int)
    feat_sizes = [(IMAGE_SIZE // 4,) * 2, (IMAGE_SIZE // 8,) * 2, (IMAGE_SIZE // 16,) * 2]

    enc_times, bspt_times, dec_times = [], [], []

    # warmup
    for _ in range(5):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bo = model.forward_image(dummy)
    torch.cuda.synchronize()

    for _ in range(num_runs):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encoder
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            backbone_out = model.forward_image(dummy)
            e.record()
            torch.cuda.synchronize()
            enc_times.append(s.elapsed_time(e))

            _, vision_feats, vision_pos_embeds, _ = model._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)
            vision_feats[-1] += torch.zeros(1, B, model.hidden_dim, device=device)
            vision_pos_embeds[-1] += torch.zeros(1, B, model.hidden_dim, device=device)
            feats = [feat.permute(1, 2, 0).view(B, -1, *fs)
                     for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]

            # BSPT
            if bspt_modules is not None:
                from func_2d.bspt_utils import apply_bspt_to_features
                s2 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                s2.record()
                image_embed, _, _ = apply_bspt_to_features(image_embed, bspt_modules)
                e2.record()
                torch.cuda.synchronize()
                bspt_times.append(s2.elapsed_time(e2))

            high_res_feats = feats[:-1]

            # Decoder
            s3 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            s3.record()
            se, de = model.sam_prompt_encoder(points=(pt, pt_label), boxes=None, masks=None, batch_size=B)
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False,
                high_res_features=high_res_feats)
            F.interpolate(low_res_masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            e3.record()
            torch.cuda.synchronize()
            dec_times.append(s3.elapsed_time(e3))

    return {
        'encoder': np.mean(enc_times),
        'bspt': np.mean(bspt_times) if bspt_times else 0,
        'decoder': np.mean(dec_times),
    }


# ============================================================
# Main
# ============================================================

def main():
    import cfg_bspt as cfg
    args = cfg.parse_args()
    for k, v in {'dafa_rank': 4, 'pfbe_scales': [3, 5],
                 'use_ula': False, 'use_dsa': False}.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    device = torch.device('cuda', args.gpu_device)

    # GPU信息
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9

    print("=" * 75)
    print("  模型效率测量 (表8 & 表9)")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"  输入: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("=" * 75)

    # ============================================
    # A. MedSAM2 Baseline
    # ============================================
    print(f"\n{'─' * 75}")
    print("  [A] MedSAM2 (Baseline)")
    print(f"{'─' * 75}")

    from func_2d.utils import get_network
    net_base = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    for name, param in net_base.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    comp_base = count_params_by_component(net_base)
    total_base = sum(c['total'] for c in comp_base.values())
    trainable_base = sum(c['trainable'] for c in comp_base.values())

    print(f"\n  参数分布:")
    print(f"  {'组件':<25} {'总参数':>12} {'可训练':>12} {'冻结':>12}")
    print(f"  {'─' * 63}")
    for cn, cc in comp_base.items():
        if cc['total'] > 0:
            print(f"  {cn:<25} {cc['total']:>12,} {cc['trainable']:>12,} {cc['total'] - cc['trainable']:>12,}")
    print(f"  {'─' * 63}")
    print(f"  {'合计':<25} {total_base:>12,} {trainable_base:>12,} {total_base - trainable_base:>12,}")
    print(f"\n  总参数: {total_base / 1e6:.2f}M, 可训练: {trainable_base / 1e6:.2f}M")

    # FLOPs - 用fvcore
    print(f"\n  FLOPs (Image Encoder)...")
    flops_enc, method = measure_flops_fvcore(net_base.image_encoder, (1, 3, IMAGE_SIZE, IMAGE_SIZE), device)
    if flops_enc is not None and flops_enc > 1000:  # 合理性检查
        print(f"  Image Encoder: {flops_enc / 1e9:.2f} GFLOPs ({method})")
        enc_flops_val = flops_enc
    else:
        print(f"  fvcore结果异常 ({flops_enc}, {method}), 使用手动估算...")
        enc_flops_val = measure_flops_manual_encoder('hiera_small')
        print(f"  Image Encoder: ~{enc_flops_val / 1e9:.1f} GFLOPs (手动估算)")

    # 推理时间
    print(f"\n  推理时间 (warmup=10, runs=50)...")
    timing_base = measure_inference_time(net_base, device=device)
    print(f"  推理时间: {timing_base['mean']:.1f} ± {timing_base['std']:.1f} ms")

    # 分段计时
    seg_base = measure_segmented_time(net_base, device=device)
    print(f"  分段: encoder={seg_base['encoder']:.1f}ms, decoder={seg_base['decoder']:.1f}ms")

    del net_base
    torch.cuda.empty_cache()

    # ============================================
    # B. BSP-MedSAM
    # ============================================
    print(f"\n{'─' * 75}")
    print("  [B] BSP-MedSAM (Ours)")
    print(f"{'─' * 75}")

    from func_2d.bspt_utils import create_bspt_modules
    net_bsp = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    for name, param in net_bsp.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    args.use_bspt = True
    args.use_dafa = True
    args.use_pfbe = True
    args.use_baam = True
    args.use_ula = False
    args.use_dsa = False
    args.dafa_rank = 4
    args.pfbe_scales = [3, 5]

    bspt_modules = create_bspt_modules(net_bsp.hidden_dim, args)
    for name, module in bspt_modules.items():
        if isinstance(module, nn.Module):
            bspt_modules[name] = module.to(device)

    total_sam2 = sum(c['total'] for c in count_params_by_component(net_bsp).values())
    bspt_detail = count_bspt_detail(bspt_modules)
    bspt_total = bspt_detail['_total']
    total_bsp = total_sam2 + bspt_total

    print(f"\n  BSP-MedSAM 新增模块:")
    print(f"  {'模块':<15} {'参数量':>12} {'占比':>8}")
    print(f"  {'─' * 37}")
    for n, p in bspt_detail.items():
        if n == '_total': continue
        pct = (p / bspt_total * 100) if bspt_total > 0 else 0
        print(f"  {n:<15} {p:>12,} {pct:>7.2f}%")
    print(f"  {'─' * 37}")
    print(f"  {'总新增':<15} {bspt_total:>12,} {'100.00%':>8}")
    print(f"\n  总参数: {total_bsp / 1e6:.2f}M (SAM2 {total_sam2 / 1e6:.2f}M + BSPT {bspt_total / 1e3:.1f}K)")

    # FLOPs - BSPT模块手动计算
    dim = net_bsp.hidden_dim  # 256
    n_tokens = (IMAGE_SIZE // 16) ** 2  # 4096

    flops_dafa = measure_flops_dafa(dim, rank=4, n_tokens=n_tokens)
    flops_pfbe = measure_flops_pfbe(n_tokens, dim)
    flops_baam = measure_flops_baam(dim, n_tokens)
    flops_bspt_total = flops_dafa + flops_pfbe + flops_baam

    print(f"\n  BSPT模块 FLOPs:")
    print(f"    DAFA:  {flops_dafa / 1e6:.1f} MFLOPs")
    print(f"    PFBE:  {flops_pfbe / 1e6:.2f} MFLOPs (parameter-free)")
    print(f"    BAAM:  {flops_baam / 1e9:.2f} GFLOPs (双向cross-attention)")
    print(f"    总计:  {flops_bspt_total / 1e9:.2f} GFLOPs")

    total_flops = enc_flops_val + flops_bspt_total
    print(
        f"\n  总FLOPs: ~{total_flops / 1e9:.1f}G (encoder {enc_flops_val / 1e9:.1f}G + BSPT {flops_bspt_total / 1e9:.2f}G)")

    # 推理时间
    print(f"\n  推理时间 (warmup=10, runs=50)...")
    timing_bsp = measure_inference_time(net_bsp, bspt_modules=bspt_modules, device=device)
    print(f"  推理时间: {timing_bsp['mean']:.1f} ± {timing_bsp['std']:.1f} ms")

    # 分段计时
    seg_bsp = measure_segmented_time(net_bsp, bspt_modules=bspt_modules, device=device)
    print(f"  分段: encoder={seg_bsp['encoder']:.1f}ms, "
          f"BSPT={seg_bsp['bspt']:.1f}ms, decoder={seg_bsp['decoder']:.1f}ms")

    del net_bsp, bspt_modules
    torch.cuda.empty_cache()

    # ============================================
    # C. 论文表格
    # ============================================
    overhead_ms = timing_bsp['mean'] - timing_base['mean']
    overhead_pct = overhead_ms / timing_base['mean'] * 100
    flops_overhead_pct = flops_bspt_total / enc_flops_val * 100

    print(f"\n\n{'=' * 75}")
    print("  表8: 模型效率对比")
    print(f"{'=' * 75}")
    print(f"  {'方法':<18} {'总参数':>8} {'新增参数':>10} {'FLOPs':>10} {'推理时间':>16}")
    print(f"  {'─' * 64}")
    print(f"  {'MedSAM2':<18} {total_base / 1e6:.2f}M{'':>5} {'—':>10} "
          f"{enc_flops_val / 1e9:.1f}G{'':>5} {timing_base['mean']:.1f}±{timing_base['std']:.1f}ms")
    print(f"  {'BSP-MedSAM':<18} {total_bsp / 1e6:.2f}M{'':>5} "
          f"{bspt_total / 1e3:.1f}K{'':>5} {total_flops / 1e9:.1f}G{'':>5} "
          f"{timing_bsp['mean']:.1f}±{timing_bsp['std']:.1f}ms")
    print(f"  {'─' * 64}")
    print(f"  开销: +{bspt_total / 1e3:.1f}K参数(+{bspt_total / total_base * 100:.2f}%), "
          f"+{flops_bspt_total / 1e9:.2f}GFLOPs(+{flops_overhead_pct:.1f}%), "
          f"+{overhead_ms:.1f}ms(+{overhead_pct:.1f}%)")

    print(f"\n  时间分段对比:")
    print(f"    {'阶段':<15} {'MedSAM2':>10} {'BSP-MedSAM':>12} {'差值':>10}")
    print(f"    {'─' * 49}")
    print(f"    {'Encoder':<15} {seg_base['encoder']:>9.1f}ms {seg_bsp['encoder']:>11.1f}ms "
          f"{seg_bsp['encoder'] - seg_base['encoder']:>+9.1f}ms")
    if seg_bsp['bspt'] > 0:
        print(f"    {'BSPT模块':<15} {'—':>10} {seg_bsp['bspt']:>11.1f}ms "
              f"{seg_bsp['bspt']:>+9.1f}ms")
    print(f"    {'Decoder':<15} {seg_base['decoder']:>9.1f}ms {seg_bsp['decoder']:>11.1f}ms "
          f"{seg_bsp['decoder'] - seg_base['decoder']:>+9.1f}ms")

    print(f"\n{'=' * 75}")
    print("  表9: BSP-MedSAM 模块参数与FLOPs分布")
    print(f"{'=' * 75}")
    print(f"  {'模块':<10} {'参数量':>10} {'参数占比':>8} {'FLOPs':>12} {'FLOPs占比':>10}")
    print(f"  {'─' * 52}")
    bspt_flops_map = {'DAFA': flops_dafa, 'PFBE': flops_pfbe, 'BAAM': flops_baam}
    for n, p in bspt_detail.items():
        if n == '_total': continue
        ppct = (p / bspt_total * 100) if bspt_total > 0 else 0
        f_val = bspt_flops_map.get(n, 0)
        fpct = (f_val / flops_bspt_total * 100) if flops_bspt_total > 0 else 0
        f_str = f"{f_val / 1e6:.1f}M" if f_val < 1e9 else f"{f_val / 1e9:.2f}G"
        print(f"  {n:<10} {p:>10,} {ppct:>7.1f}% {f_str:>12} {fpct:>9.1f}%")
    print(f"  {'─' * 52}")
    print(f"  {'总计':<10} {bspt_total:>10,} {'100.0%':>8} {flops_bspt_total / 1e9:.2f}G{'':>5} {'100.0%':>10}")
    print(f"{'=' * 75}")

    print(f"\n  GPU: {gpu_name}")
    print(f"  注: FLOPs为理论值 (encoder用{'fvcore' if method == 'fvcore' else '手动估算'}, "
          f"BSPT用手动计算)")


if __name__ == '__main__':
    main()