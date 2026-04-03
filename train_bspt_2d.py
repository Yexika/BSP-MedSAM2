#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
BSPT-MedSAM 2D 训练脚本 (重构版)
================================
train_bspt_2d.py
保持与 train_2d_improved.py 相同的代码结构
只添加 BSPT 模块集成
主要改动:
1. 使用 cfg_2d 配置
2. 集成 BSPT 模块和 HBALLoss
3. 保持原有的学习率调度和训练流程
"""
import os
import time
import math
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import cfg_bspt as cfg
from conf import settings
from func_2d.dataset import REFUGE
from func_2d.dataset_modified import MultiDataset
from func_2d.utils import get_network, set_log_dir, create_logger

# ★★★ BSPT 导入 ★★★
from func_2d.bspt_utils import create_bspt_modules, create_bspt_loss
import func_2d.function_bspt as function  # 使用集成了BSPT的function

#
# 基础路径
BASE_PATH = '/root/autodl-tmp/BSPT-Medsam'
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight')
DATA_ROOT = '/root/autodl-tmp/datasets'

SUPPORTED_DATASETS = [
    'CVC-ClinicDB',
    'ISIC17',
    'ISIC18',
    'Kvasir-SEG',#这两个数据集已经训练完成
    'DSB18'
]
# 'ETIS-LaribPolypDB',
# 'BUSI',

# ============== 学习率调度器 ==============

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_ratio=0.01, num_cycles=0.5):
    """
    带预热的余弦退火调度器

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率比例 (相对于初始lr)
        num_cycles: 余弦周期数
    """

    def lr_lambda(current_step):
        # Warmup 阶段: 线性增加
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine 退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # 确保不低于最小学习率
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def train_single_dataset(args, dataset_name, data_path):
    """训练单个数据集 (BSPT 版本)"""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    GPUdevice = torch.device('cuda', args.gpu_device)
    os.makedirs(WEIGHT_PATH, exist_ok=True)

    # ============== 权重保存路径 ==============
    ablation_name = getattr(args, 'ablation_name', '')
    if ablation_name:
        ablation_weight_dir = os.path.join(WEIGHT_PATH, 'ablation', ablation_name)
        os.makedirs(ablation_weight_dir, exist_ok=True)
        weight_save_path = os.path.join(ablation_weight_dir, f'BSPT_MedSAM_{dataset_name}.pth')
    else:
        weight_save_path = os.path.join(WEIGHT_PATH, f'BSPT_MedSAM_{dataset_name}.pth')

    # ============== 训练超参数 ==============
    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    min_lr_ratio = getattr(args, 'min_lr_ratio', 0.01)
    max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    use_bspt = getattr(args, 'use_bspt', True)

    print("=" * 70)
    print(f"BSPT-MedSAM 2D 训练 - {dataset_name}")
    if ablation_name:
        print(f"消融实验: {ablation_name}")
    print("=" * 70)
    print(f"数据路径: {data_path}")
    print(f"图像尺寸: {args.image_size} x {args.image_size}")
    print(f"批量大小: {args.b}")
    print(f"初始学习率: {args.lr}")
    print(f"预热 Epochs: {warmup_epochs}")
    print(f"最小学习率比例: {min_lr_ratio}")
    print(f"梯度裁剪阈值: {max_grad_norm}")
    print(f"权重衰减: {weight_decay}")
    print(f"BSPT 模块: {'启用' if use_bspt else '禁用'}")
    print(f"权重保存路径: {weight_save_path}")
    print("=" * 70)

    # 初始化网络
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    """

    # ★★★ 加载 MedSAM2 预训练权重 ★★★
    pretrain_path = getattr(args, 'pretrain', None) or getattr(args, 'weights', None)
    if pretrain_path and os.path.exists(pretrain_path):
        print(f"\n加载 MedSAM2 预训练权重: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=GPUdevice)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']

        model_dict = net.state_dict()
        # 过滤：只加载 key 匹配且 shape 匹配的参数
        matched = {k: v for k, v in checkpoint.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        skipped = [k for k in checkpoint.keys() if k not in matched]
        missing = [k for k in model_dict.keys() if k not in checkpoint]

        model_dict.update(matched)
        net.load_state_dict(model_dict)

        print(f"  已加载: {len(matched)}/{len(checkpoint)} 参数")
        if skipped:
            print(f"  跳过 (不匹配): {len(skipped)} 个 key")
        if missing:
            print(f"  模型中未覆盖: {len(missing)} 个 key")
        print(f"  MedSAM2 预训练权重加载完成!\n")
    else:
        print(f"\n⚠️  未找到 MedSAM2 预训练权重: {pretrain_path}")
        print("  将使用 SAM2 原始权重作为起点\n")
    """

    #
    # ★★★ 初始化 BSPT 模块 ★★★
    if use_bspt:
        print("\n🚀 初始化 BSPT 模块...")
        bspt_modules = create_bspt_modules(net.hidden_dim, args)
        for name, module in bspt_modules.items():
            bspt_modules[name] = module.to(GPUdevice)

        # 创建 BSPT 损失函数
        bspt_loss = create_bspt_loss(args, GPUdevice)

        # 将模块和损失存储到 args 中
        args._bspt_modules = bspt_modules
        args._bspt_loss = bspt_loss
        print("✅ BSPT 模块初始化完成\n")

    # ★★★ 优化器 - 分离 BSPT 参数和基础模型参数 ★★★
    base_params = []
    bspt_params = []

    for name, param in net.named_parameters():
        if param.requires_grad:
            base_params.append(param)

    if use_bspt:
        for module in args._bspt_modules.values():
            for param in module.parameters():
                if param.requires_grad:
                    bspt_params.append(param)

    # 分组优化器参数
    param_groups = [
        {'params': base_params, 'lr': args.lr},
    ]

    if bspt_params:
        param_groups.append({'params': bspt_params, 'lr': args.lr})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # 日志设置
    log_path = os.path.join('logs', f'{args.exp_name}_BSPT_{dataset_name}')
    args.path_helper = set_log_dir(log_path, args.exp_name)
    logger = create_logger(args.path_helper['log_path'])

    logger.handlers = logger.handlers[-1:]  # ★ 只保留最新的 handler

    logger.info(f"训练数据集: {dataset_name}")
    if ablation_name:
        logger.info(f"消融实验: {ablation_name}")
    logger.info(args)

    # 数据变换
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    if dataset_name == 'REFUGE':
        train_dataset = REFUGE(args, data_path, transform=transform_train, mode='Training')
        test_dataset = REFUGE(args, data_path, transform=transform_test, mode='Test')
    else:
        train_dataset = MultiDataset(args, data_path, mode='Training', prompt='click', seed=args.seed)
        test_dataset = MultiDataset(args, data_path, mode='Test', prompt='click', seed=args.seed)

    nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")

    # ★★★ 计算总训练步数和预热步数 ★★★
    total_epochs = getattr(args, 'epochs', settings.EPOCH)
    steps_per_epoch = len(nice_train_loader)
    total_training_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    print(f"每 Epoch 步数: {steps_per_epoch}")
    print(f"总训练步数: {total_training_steps}")
    print(f"预热步数: {warmup_steps}")

    # ★★★ 创建学习率调度器 ★★★
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=min_lr_ratio
    )

    # TensorBoard - 消融实验保存到 ablation 子目录
    if ablation_name:
        log_dir = os.path.join(settings.LOG_DIR, args.net,
                               f'ablation_{ablation_name}_{dataset_name}_{settings.TIME_NOW}')
    else:
        log_dir = os.path.join(settings.LOG_DIR, args.net, f'BSPT_{dataset_name}_{settings.TIME_NOW}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_dice = 0.0
    best_metrics = {}
    global_step = 0
    no_improve_count = 0
    early_stop_patience = getattr(args, 'early_stop_patience', 0)

    for epoch in range(total_epochs):
        # 初始验证
        if epoch == 0:
            net.eval()
            tol, metrics_tuple = function.validation_sam(args, nice_test_loader, epoch, net)
            avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd = metrics_tuple
            logger.info(f'[Epoch {epoch}] 初始 - 平均Dice: {avg_dice:.4f}, 平均IoU: {avg_iou:.4f}')

        # ★★★ 训练 (使用 BSPT 集成的训练函数) ★★★
        net.train()

        # 设置 BSPT 模块为训练模式
        if use_bspt:
            for module in args._bspt_modules.values():
                module.train()

        time_start = time.time()

        loss, global_step = function.train_sam_with_scheduler(
            args, net, optimizer, nice_train_loader, epoch, writer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            global_step=global_step
        )

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)

        logger.info(f'训练损失: {loss:.4f} @ epoch {epoch}, lr: {current_lr:.2e}')
        print(f'训练耗时: {time.time() - time_start:.2f}秒, 当前学习率: {current_lr:.2e}')

        # 验证
        net.eval()

        # 设置 BSPT 模块为评估模式
        if use_bspt:
            for module in args._bspt_modules.values():
                module.eval()

        if epoch % args.val_freq == 0 or epoch == total_epochs - 1:
            tol, metrics_tuple = function.validation_sam(args, nice_test_loader, epoch, net)
            avg_iou, avg_dice, global_dice, global_iou, avg_hd95, avg_asd, global_hd95, global_asd = metrics_tuple

            logger.info(f'[Epoch {epoch}] 平均Dice: {avg_dice:.4f}, 平均IoU: {avg_iou:.4f}, '
                        f'平均HD95: {avg_hd95:.2f}, 平均ASD: {avg_asd:.2f}')

            writer.add_scalar('val/avg_dice', avg_dice, epoch)
            writer.add_scalar('val/avg_iou', avg_iou, epoch)
            writer.add_scalar('val/avg_hd95', avg_hd95, epoch)

            # 保存最佳模型
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_metrics = {
                    'epoch': epoch, 'avg_dice': avg_dice, 'avg_iou': avg_iou,
                    'avg_hd95': avg_hd95, 'avg_asd': avg_asd,
                    'global_dice': global_dice, 'global_iou': global_iou,
                    'global_hd95': global_hd95, 'global_asd': global_asd
                }

                # 保存模型和BSPT模块
                save_dict = {
                    'model': net.state_dict(),
                    'epoch': epoch,
                    'metrics': best_metrics,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                if use_bspt:
                    save_dict['bspt_modules'] = {
                        name: module.state_dict()
                        for name, module in args._bspt_modules.items()
                    }

                torch.save(save_dict, weight_save_path)
                logger.info(f'*** 最佳模型已保存! Dice: {best_dice:.4f} @ {weight_save_path} ***')
                no_improve_count = 0
            else:
                no_improve_count += 1
                if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
                    logger.info(f'⏹ 早停: {early_stop_patience} epochs 无提升, '
                                f'最佳Dice: {best_dice:.4f} @ epoch {best_metrics.get("epoch", "N/A")}')
                    break

    logger.info("=" * 70)
    logger.info(
        f"训练完成! 最佳Epoch: {best_metrics.get('epoch', 'N/A')}, 最佳Dice: {best_metrics.get('avg_dice', 0):.4f}")
    logger.info("=" * 70)
    writer.close()

    return best_metrics


def main():
    args = cfg.parse_args()
    # ★★★ 添加过滤参数 ★★★
    if not hasattr(args, 'filter_area_threshold'):
        args.filter_area_threshold = 0.8  # 预测区域>90%视为异常
    if not hasattr(args, 'filter_min_dice'):
        args.filter_min_dice = 0.50  # Dice<0.05且预测区域大视为异常
    # ★★★ 添加 BSPT 参数默认值 ★★★
    if not hasattr(args, 'use_bspt'):
        args.use_bspt = True
    if not hasattr(args, 'use_dafa'):
        args.use_dafa = True
    if not hasattr(args, 'use_pfbe'):
        args.use_pfbe = True
    if not hasattr(args, 'use_baam'):
        args.use_baam = True
    if not hasattr(args, 'use_ula'):
        args.use_ula = True
    if not hasattr(args, 'use_dsa'):
        args.use_dsa = True

    if not hasattr(args, 'dafa_rank'):
        args.dafa_rank = 4
    if not hasattr(args, 'ula_compression'):
        args.ula_compression = 16
    if not hasattr(args, 'dsa_sparsity'):
        args.dsa_sparsity = 0.25
    if not hasattr(args, 'pfbe_scales'):
        args.pfbe_scales = [3, 5 ,7]

    if not hasattr(args, 'lambda_dice'):
        args.lambda_dice = 1.0
    if not hasattr(args, 'lambda_bce'):
        args.lambda_bce = 0.5
    if not hasattr(args, 'lambda_boundary'):
        args.lambda_boundary = 0.3
    if not hasattr(args, 'lambda_hd'):
        args.lambda_hd = 0.1
    if not hasattr(args, 'hbal_warmup'):
        args.hbal_warmup = 10

    # 添加学习率调度参数默认值
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 10
    if not hasattr(args, 'min_lr_ratio'):
        args.min_lr_ratio = 0.01
    if not hasattr(args, 'max_grad_norm'):
        args.max_grad_norm = 1.0
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 0.01

    # 确定要训练的数据集
    if args.dataset.lower() == 'all':
        datasets_to_train = SUPPORTED_DATASETS
        print(f"\n将训练全部 {len(datasets_to_train)} 个数据集: {datasets_to_train}\n")
    else:
        datasets_to_train = [args.dataset]
        print(f"\n将训练单个数据集: {args.dataset}\n")

    all_results = {}

    for dataset_name in datasets_to_train:
        if args.data_path and args.dataset.lower() != 'all':
            data_path = args.data_path
        else:
            data_path = os.path.join(DATA_ROOT, dataset_name)

        if not os.path.exists(data_path):
            print(f"警告: 数据集路径不存在，跳过 {dataset_name}: {data_path}")
            continue

        print(f"\n{'#' * 70}")
        print(f"# 开始训练: {dataset_name}")
        print(f"{'#' * 70}\n")

        try:
            best_metrics = train_single_dataset(args, dataset_name, data_path)
            all_results[dataset_name] = best_metrics
        except Exception as e:
            print(f"训练 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印汇总结果
    print("\n" + "=" * 70)
    print("BSPT-MedSAM 训练完成! 汇总结果:")
    print("=" * 70)
    for ds_name, metrics in all_results.items():
        print(f"{ds_name}: Dice={metrics.get('avg_dice', 0):.4f}, IoU={metrics.get('avg_iou', 0):.4f}, "
              f"HD95={metrics.get('avg_hd95', 'inf'):.2f}, ASD={metrics.get('avg_asd', 'inf'):.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()