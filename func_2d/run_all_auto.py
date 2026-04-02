#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
run_all_auto.py - 全自动训练+测试+效率测量
==========================================
出门前启动，回来收结果。

执行顺序:
  Phase 1: 训练3个新数据集 (CVC-ClinicDB, ISIC17, ISIC18)
           Kvasir-SEG和DSB18的E1权重已有，自动跳过
  Phase 2: 超参数消融 - DAFA rank (表6, 4个实验)
  Phase 3: 超参数消融 - HBAL weights (表7, 4个实验)
  Phase 4: 测量模型效率 (表8/9)
  Phase 5: 测试超参数结果

用法:
  cd /root/autodl-tmp/BSPT-Medsam
  nohup python run_all_auto.py > auto_train.log 2>&1 &
  tail -f auto_train.log
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

BASE_PATH = '/root/autodl-tmp/BSPT-Medsam'
sys.path.insert(0, BASE_PATH)
DATA_ROOT = '/root/autodl-tmp/datasets'
WEIGHT_PATH = os.path.join(BASE_PATH, 'weight')

# 只需训练这3个（Kvasir-SEG和DSB18已有权重）
DATASETS_TO_TRAIN = ['CVC-ClinicDB', 'ISIC17', 'ISIC18']

LOG_FILE = os.path.join(BASE_PATH, 'auto_progress.log')


def log(msg):
    ts = datetime.now().strftime('%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_cmd(cmd, desc, timeout_hours=4):
    log(f"  CMD: {cmd}")
    start = time.time()
    try:
        result = subprocess.run(cmd, shell=True, cwd=BASE_PATH, timeout=timeout_hours*3600)
        elapsed = (time.time() - start) / 60
        ok = result.returncode == 0
        log(f"  {'✅' if ok else '❌'} {desc} ({elapsed:.0f}min)")
        return ok
    except subprocess.TimeoutExpired:
        log(f"  ⏰ 超时: {desc}")
        return False
    except Exception as e:
        log(f"  ❌ 异常: {e}")
        return False


# ============================================================
def phase1_train_datasets():
    log("=" * 60)
    log("PHASE 1: 训练3个新数据集")
    log("=" * 60)

    results = {}
    for ds in DATASETS_TO_TRAIN:
        weight_file = os.path.join(WEIGHT_PATH, f'BSPT_MedSAM_{ds}.pth')

        if os.path.exists(weight_file):
            log(f"  ⏭️  {ds}: 权重已存在，跳过")
            results[ds] = 'skipped'
            continue

        data_path = os.path.join(DATA_ROOT, ds)
        if not os.path.exists(data_path):
            log(f"  ❌ {ds}: 数据集不存在 {data_path}")
            results[ds] = 'missing'
            continue

        log(f"\n  🏋️ 训练: {ds}")
        ok = run_cmd(
            f"python train_bspt_2d.py --dataset {ds} --data_path {data_path}",
            f"训练 {ds}", timeout_hours=4
        )
        results[ds] = 'done' if ok else 'failed'

    log(f"\n  Phase 1 汇总:")
    for ds, st in results.items():
        log(f"    {'✅' if st in ('done','skipped') else '❌'} {ds}: {st}")
    return results


def phase2_dafa_rank():
    log("=" * 60)
    log("PHASE 2: DAFA rank消融 (表6)")
    log("=" * 60)

    script = os.path.join(BASE_PATH, 'run_hyperparam_sweep.py')
    if not os.path.exists(script):
        log(f"  ❌ 脚本不存在: {script}")
        return False
    return run_cmd("python run_hyperparam_sweep.py --table 6",
                    "DAFA rank sweep", timeout_hours=6)


def phase3_hbal_weights():
    log("=" * 60)
    log("PHASE 3: HBAL权重消融 (表7)")
    log("=" * 60)

    script = os.path.join(BASE_PATH, 'run_hyperparam_sweep.py')
    if not os.path.exists(script):
        log(f"  ❌ 脚本不存在: {script}")
        return False
    return run_cmd("python run_hyperparam_sweep.py --table 7",
                    "HBAL weight sweep", timeout_hours=6)


def phase4_efficiency():
    log("=" * 60)
    log("PHASE 4: 模型效率测量 (表8/9)")
    log("=" * 60)

    script = os.path.join(BASE_PATH, 'measure_efficiency.py')
    if not os.path.exists(script):
        log(f"  ❌ 脚本不存在: {script}")
        return False
    return run_cmd("python measure_efficiency.py", "效率测量", timeout_hours=0.5)


def phase5_test():
    log("=" * 60)
    log("PHASE 5: 测试全部结果")
    log("=" * 60)

    # 超参数测试
    test_sweep = os.path.join(BASE_PATH, 'test_hyperparam_sweep.py')
    if os.path.exists(test_sweep):
        run_cmd("python test_hyperparam_sweep.py", "超参数测试", timeout_hours=1)

    return True


# ============================================================
def main():
    t0 = time.time()

    log("\n" + "=" * 60)
    log("  BSP-MedSAM 全自动流水线")
    log(f"  开始: {datetime.now():%Y-%m-%d %H:%M}")
    log(f"  待训练: {DATASETS_TO_TRAIN}")
    log("=" * 60)

    # 检查已有权重
    log("\n  已有权重:")
    for ds in ['Kvasir-SEG', 'DSB18'] + DATASETS_TO_TRAIN:
        w = os.path.join(WEIGHT_PATH, f'BSPT_MedSAM_{ds}.pth')
        log(f"    {ds}: {'✅ 存在' if os.path.exists(w) else '❌ 需训练'}")

    phases = {}

    try:
        phases['P1_train'] = phase1_train_datasets()
    except Exception as e:
        log(f"  Phase 1 异常: {e}")
        phases['P1_train'] = str(e)

    try:
        phases['P2_dafa'] = phase2_dafa_rank()
    except Exception as e:
        log(f"  Phase 2 异常: {e}")
        phases['P2_dafa'] = False

    try:
        phases['P3_hbal'] = phase3_hbal_weights()
    except Exception as e:
        log(f"  Phase 3 异常: {e}")
        phases['P3_hbal'] = False

    try:
        phases['P4_eff'] = phase4_efficiency()
    except Exception as e:
        log(f"  Phase 4 异常: {e}")
        phases['P4_eff'] = False

    try:
        phases['P5_test'] = phase5_test()
    except Exception as e:
        log(f"  Phase 5 异常: {e}")
        phases['P5_test'] = False

    # 汇总
    hours = (time.time() - t0) / 3600
    log("\n" + "=" * 60)
    log(f"  全部完成! 耗时: {hours:.1f}小时")
    log(f"  {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 60)

    for phase, result in phases.items():
        if isinstance(result, dict):
            n_ok = sum(1 for v in result.values() if v in ('done', 'skipped'))
            n_fail = sum(1 for v in result.values() if v == 'failed')
            log(f"  {'✅' if n_fail==0 else '❌'} {phase}: {n_ok}ok/{n_fail}fail")
        else:
            log(f"  {'✅' if result else '❌'} {phase}")

    # 列出权重
    log("\n  权重文件:")
    for root, dirs, files in os.walk(WEIGHT_PATH):
        for f in sorted(files):
            if f.endswith('.pth'):
                p = os.path.join(root, f)
                mb = os.path.getsize(p) / 1024 / 1024
                log(f"    {os.path.relpath(p, WEIGHT_PATH)} ({mb:.0f}MB)")

    log("\n  🎉 Done!")


if __name__ == '__main__':
    main()