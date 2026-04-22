# -*- coding: utf-8 -*-
"""
数据集整理脚本 - 将采集的数据拆分并放入训练目录

运行前提：已用 collect_data.py 采集了足够的图片

功能：
  1. 读取 collected_data/ 下的所有图片和标注
  2. 按 70% / 20% / 10% 随机拆分为 train / valid / test
  3. 复制到 Drowsiness/images/ 和 Drowsiness/labels/ 对应子目录
  4. 打印最终各集合的样本统计

目标目录结构（与 drowsiness.yaml 一致）：
  Drowsiness/
  ├── images/
  │   ├── train/
  │   ├── valid/
  │   └── test/
  └── labels/
      ├── train/
      ├── valid/
      └── test/
"""

import random
import shutil
from pathlib import Path

# ──────────────────── 配置 ────────────────────
COLLECT_DIR = Path("collected_data")     # collect_data.py 的输出目录
OUTPUT_DIR  = Path("Drowsiness")         # 项目训练数据目录

TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
# TEST_RATIO  = 1 - TRAIN - VALID = 0.10

RANDOM_SEED = 42                         # 固定随机种子，保证可复现
# ─────────────────────────────────────────────

CLASS_NAMES = ['Eyeclosed', 'Neutral', 'Yawn']


def setup_output_dirs():
    for split in ('train', 'valid', 'test'):
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)


def collect_samples():
    """
    收集所有 (image_path, label_path) 对。
    只保留同时有图片和标注文件的样本。
    """
    samples = []
    missing_labels = 0

    for class_name in CLASS_NAMES:
        img_dir   = COLLECT_DIR / 'images' / class_name
        label_dir = COLLECT_DIR / 'labels' / class_name

        if not img_dir.exists():
            print(f"[警告] 未找到目录: {img_dir}，跳过 {class_name}")
            continue

        for img_path in sorted(img_dir.glob('*.jpg')):
            label_path = label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                samples.append((img_path, label_path))
            else:
                missing_labels += 1
                print(f"[警告] 缺少标注文件: {label_path}，跳过")

    if missing_labels:
        print(f"共跳过 {missing_labels} 个缺少标注的样本")

    return samples


def split_samples(samples):
    """随机打乱后按比例拆分"""
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train = samples[:n_train]
    valid = samples[n_train:n_train + n_valid]
    test  = samples[n_train + n_valid:]

    return train, valid, test


def safe_copy(src: Path, dst_dir: Path):
    """
    复制文件到目标目录，若同名文件已存在则自动重命名（加数字后缀），
    绝不覆盖已有文件。
    """
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return
    # 已存在同名文件，找一个不冲突的新名字
    stem, suffix = src.stem, src.suffix
    counter = 1
    while True:
        dst = dst_dir / f"{stem}_dup{counter}{suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
            return
        counter += 1


def copy_samples(samples, split_name):
    """将样本复制到对应的 split 目录，不覆盖已有文件"""
    img_dst   = OUTPUT_DIR / 'images' / split_name
    label_dst = OUTPUT_DIR / 'labels' / split_name

    for img_path, label_path in samples:
        safe_copy(img_path,   img_dst)
        safe_copy(label_path, label_dst)


def main():
    # 检查源目录
    if not COLLECT_DIR.exists():
        print(f"[错误] 找不到采集目录: {COLLECT_DIR.resolve()}")
        print("请先运行 collect_data.py 采集数据。")
        return

    samples = collect_samples()
    if len(samples) == 0:
        print("[错误] 没有找到任何有效样本（图片+标注都存在）。")
        return

    print(f"\n共找到 {len(samples)} 个有效样本")

    # 统计各类别数量
    class_counts = {c: 0 for c in CLASS_NAMES}
    for img_path, _ in samples:
        class_counts[img_path.parent.name] += 1
    for name, cnt in class_counts.items():
        print(f"  {name}: {cnt} 张")

    # 建议最少样本数
    min_count = min(class_counts.values())
    if min_count < 100:
        print(f"\n[建议] {min(class_counts, key=class_counts.get)} 类样本较少（{min_count} 张），"
              f"建议每类至少采集 100 张以上，训练效果更好。")

    # 拆分
    train, valid, test = split_samples(samples)
    print(f"\n拆分结果:")
    print(f"  train: {len(train)} 张")
    print(f"  valid: {len(valid)} 张")
    print(f"  test:  {len(test)} 张")

    # 询问是否继续（避免误操作覆盖已有数据）
    existing = list((OUTPUT_DIR / 'images' / 'train').glob('*.jpg'))
    if existing:
        ans = input(f"\n[警告] {OUTPUT_DIR}/images/train 已有 {len(existing)} 张图片，"
                    f"继续将追加/覆盖。继续？(y/n): ").strip().lower()
        if ans != 'y':
            print("已取消。")
            return

    # 创建输出目录并复制
    setup_output_dirs()
    copy_samples(train, 'train')
    copy_samples(valid, 'valid')
    copy_samples(test,  'test')

    print("\n整理完成！目录结构：")
    for split in ('train', 'valid', 'test'):
        n = len(list((OUTPUT_DIR / 'images' / split).glob('*.jpg')))
        print(f"  Drowsiness/images/{split}/  ← {n} 张")

    print(f"\n训练命令：")
    print(f"  python train.py --data Drowsiness/drowsiness.yaml "
          f"--weights yolov5s.pt --img 640 --epochs 100 --batch-size 16")


if __name__ == "__main__":
    main()
