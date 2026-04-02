"""
func_2d/function1.py - MedSAM2 2D 训练/验证函数
- 计算指标: Dice, IoU, HD95, ASD (全局 & 平均)
- 蓝色半透明可视化 (只保存单张预测结果图)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import pandas as pd

import cfg_2d as cfg
from conf import settings
from func_2d.utils import *

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32

torch.backends.cudnn.benchmark = True


def get_feat_sizes(image_size):
    """计算特征图尺寸"""
    return [
        (image_size // 4, image_size // 4),
        (image_size // 8, image_size // 8),
        (image_size // 16, image_size // 16)
    ]


# ============== 指标计算函数 ==============

def compute_dice(pred, gt):
    """计算Dice系数"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-6)


def compute_iou(pred, gt):
    """计算IoU"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-6)


def compute_surface_distances(pred, gt, spacing=(1.0, 1.0)):
    """计算表面距离"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return None, None

    pred_border = pred ^ ndimage.binary_erosion(pred)
    gt_border = gt ^ ndimage.binary_erosion(gt)

    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return None, None

    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    pred_to_gt = dt_gt[pred_border]
    gt_to_pred = dt_pred[gt_border]

    return pred_to_gt, gt_to_pred


def compute_hd95(pred, gt, spacing=(1.0, 1.0)):
    """计算95% Hausdorff距离"""
    d_p2g, d_g2p = compute_surface_distances(pred, gt, spacing)
    if d_p2g is None or d_g2p is None:
        return float('inf')
    all_distances = np.concatenate([d_p2g, d_g2p])
    return np.percentile(all_distances, 95)


def compute_asd(pred, gt, spacing=(1.0, 1.0)):
    """计算平均表面距离"""
    d_p2g, d_g2p = compute_surface_distances(pred, gt, spacing)
    if d_p2g is None or d_g2p is None:
        return float('inf')
    return (np.mean(d_p2g) + np.mean(d_g2p)) / 2


# ============== 可视化函数 ==============

def save_blue_overlay(img, pred, save_path, alpha=0.5):
    """
    保存蓝色半透明覆盖的可视化结果（只保存单张图）
    """
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)

    pred_np = pred.cpu().numpy().squeeze()

    overlay = img_np.copy().astype(np.float32)

    pred_mask = pred_np > 0.5
    overlay[pred_mask, 0] = overlay[pred_mask, 0] * (1 - alpha) + 50 * alpha
    overlay[pred_mask, 1] = overlay[pred_mask, 1] * (1 - alpha) + 100 * alpha
    overlay[pred_mask, 2] = overlay[pred_mask, 2] * (1 - alpha) + 255 * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(save_path)


# ============== 训练函数 ==============

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer):
    """训练函数"""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G

    feat_sizes = get_feat_sizes(args.image_size)
    embed_size = args.image_size // 16

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=mask_type, device=GPUdevice)

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)

            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
            else:
                for element in memory_bank_list:
                    to_cat_memory.append(element[0].cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_memory_pos.append(element[1].cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                    to_cat_image_embed.append(element[3].cuda(non_blocking=True))

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, embed_size, embed_size)
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                similarity_scores = F.softmax(similarity_scores, dim=1)
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                memory_stack_ori_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
                memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                    memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
                )

            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            with torch.no_grad():
                #flag = (ind % 5) == 0
                #points = (coords_torch, labels_torch) if flag else None
                flag = True
                points = (coords_torch, labels_torch)
                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

            low_res_multimasks, iou_predictions, _, _ = net.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                multimask_output=False, repeat_image=False, high_res_features=high_res_feats
            )

            pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)

            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks, is_mask_from_pts=flag)

            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([
                        maskmem_features[batch].unsqueeze(0).detach(),
                        maskmem_pos_enc[batch].unsqueeze(0).detach(),
                        iou_predictions[batch, 0],
                        image_embed[batch].reshape(-1).detach()
                    ])
            else:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t())
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores)
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index)
                            memory_bank_list.append([
                                maskmem_features[batch].unsqueeze(0).detach(),
                                maskmem_pos_enc[batch].unsqueeze(0).detach(),
                                iou_predictions[batch, 0],
                                image_embed[batch].reshape(-1).detach()
                            ])

            loss = lossfunc(pred, masks)
            pbar.set_postfix(**{'loss': loss.item()})
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update()

    return epoch_loss / len(train_loader)


# ============== 验证函数 ==============

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    """
    验证函数 - 计算完整指标并返回
    返回: (loss, (avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd))
    """
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net.eval()
    n_val = len(val_loader)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    lossfunc = criterion_G
    memory_bank_list = []
    feat_sizes = get_feat_sizes(args.image_size)
    embed_size = args.image_size // 16

    total_loss = 0
    per_sample_results = []

    # 全局累加器
    total_intersection = 0
    total_union = 0
    total_pred_sum = 0
    total_gt_sum = 0
    all_pred_to_gt = []
    all_gt_to_pred = []

    with tqdm(total=n_val, desc='验证中', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice)

            if 'pt' in pack:
                pt_temp = pack['pt'].to(device=GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device=GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            with torch.no_grad():
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                else:
                    for element in memory_bank_list:
                        to_cat_memory.append(element[0].cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(element[1].cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append(element[3].cuda(non_blocking=True))

                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, embed_size, embed_size)
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                    similarity_scores = F.softmax(similarity_scores, dim=1)
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                    memory_stack_ori_new = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
                    memory_pos_stack_new = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3)
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]], curr_pos=[vision_pos_embeds[-1]],
                        memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                #flag = (ind % 5) == 0
                #points = (coords_torch, labels_torch) if flag else None
                flag = True
                points = (coords_torch, labels_torch)
                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

                low_res_multimasks, iou_predictions, _, _ = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                    multimask_output=False, repeat_image=False, high_res_features=high_res_feats
                )

                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks, is_mask_from_pts=flag)

                maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

                # 更新记忆库
                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([
                            maskmem_features[batch].unsqueeze(0),
                            maskmem_pos_enc[batch].unsqueeze(0),
                            iou_predictions[batch, 0],
                            image_embed[batch].reshape(-1).detach()
                        ])

                total_loss += lossfunc(pred, masks)
                pred_binary = (torch.sigmoid(pred) > 0.5).float()

                # 计算每个样本的指标
                for b in range(B):
                    sample_name = name[b] if isinstance(name, (list, tuple)) else name
                    pred_np = pred_binary[b, 0].cpu().numpy()
                    gt_np = masks[b, 0].cpu().numpy()

                    dice = compute_dice(pred_np, gt_np)
                    iou_val = compute_iou(pred_np, gt_np)
                    hd95 = compute_hd95(pred_np, gt_np)
                    asd = compute_asd(pred_np, gt_np)

                    per_sample_results.append({
                        'name': sample_name, 'dice': dice, 'iou': iou_val,
                        'hd95': hd95 if not np.isinf(hd95) else -1,
                        'asd': asd if not np.isinf(asd) else -1
                    })

                    # 全局累加
                    pred_flat = pred_np.flatten().astype(bool)
                    gt_flat = gt_np.flatten().astype(bool)
                    total_intersection += np.sum(pred_flat & gt_flat)
                    total_union += np.sum(pred_flat | gt_flat)
                    total_pred_sum += np.sum(pred_flat)
                    total_gt_sum += np.sum(gt_flat)

                    d_p2g, d_g2p = compute_surface_distances(pred_np, gt_np)
                    if d_p2g is not None and d_g2p is not None:
                        all_pred_to_gt.extend(d_p2g.tolist())
                        all_gt_to_pred.extend(d_g2p.tolist())

            pbar.update()

    # 计算全局指标
    smooth = 1e-6
    global_dice = (2 * total_intersection + smooth) / (total_pred_sum + total_gt_sum + smooth)
    global_iou = (total_intersection + smooth) / (total_union + smooth)

    if len(all_pred_to_gt) > 0 and len(all_gt_to_pred) > 0:
        all_distances = np.array(all_pred_to_gt + all_gt_to_pred)
        global_hd95 = np.percentile(all_distances, 95)
        global_asd = (np.mean(all_pred_to_gt) + np.mean(all_gt_to_pred)) / 2
    else:
        global_hd95 = float('inf')
        global_asd = float('inf')

    # 计算平均指标
    valid_hd95 = [r['hd95'] for r in per_sample_results if r['hd95'] >= 0]
    valid_asd = [r['asd'] for r in per_sample_results if r['asd'] >= 0]

    avg_dice = np.mean([r['dice'] for r in per_sample_results])
    avg_iou = np.mean([r['iou'] for r in per_sample_results])
    avg_hd95 = np.mean(valid_hd95) if valid_hd95 else float('inf')
    avg_asd = np.mean(valid_asd) if valid_asd else float('inf')

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} 验证结果")
    print(f"{'='*60}")
    print(f"[全局指标] Dice: {global_dice:.4f} | IoU: {global_iou:.4f} | HD95: {global_hd95:.2f} | ASD: {global_asd:.2f}")
    print(f"[平均指标] Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | HD95: {avg_hd95:.2f} | ASD: {avg_asd:.2f}")
    print(f"{'='*60}\n")

    return total_loss / n_val, (avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd)