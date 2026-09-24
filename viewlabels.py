#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_yolo_direct_config.py

直接在代码里配置参数（不使用命令行参数）：
- 支持递归查找 images 目录下图片
- 标签为 YOLO 归一化格式（class x_center y_center width height），可能包含额外字段但忽略
- labels 目录中未匹配到图片的 .txt 会写入 unmatched_labels.txt（在 OUT_DIR 下）
- 可配置是否跳过无标签图片（SKIP_NO_LABEL），不显示类别文字、置信度或透明填充
"""

import os
from glob import glob
import cv2

# ================== 在这里配置（修改这部分即可） ==================
# IMAGES_DIR = "output/MBM-YOLO11/INbreast_640_heatmap"        # 图片根目录（会递归查找）
# LABELS_DIR = "INbreast_640/val/labels"        # label .txt 所在目录（同名匹配）
# OUT_DIR = "output/MBM-YOLO11/INbreast_640_heatmap_vis"       # 可视化结果输出目录（会自动创建）
IMAGES_DIR = "heatmap/INbreast_3000/train/images"        # 图片根目录（会递归查找）
LABELS_DIR = "heatmap/INbreast_3000/train/labels"  # label .txt 所在目录（同名匹配）
OUT_DIR = "heatmap/INbreast_3000/vis"  # 可视化结果输出目录（会自动创建）
CLASSNAMES_PATH = None      # 可选：类名文件路径（每行一个类名），若不需要设为 None（此脚本不显示名称，仅保留占位）
SHOW_CONF = False           # 保留配置项但脚本不会显示置信度（此处无效）
SKIP_NO_LABEL = False       # 是否跳过没有 label 的图片（True=跳过，不保存）
IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp"]  # 搜索的图片后缀（小写）
BBOX_COLOR = (0, 255, 0)    # 默认 bbox 颜色 (B,G,R)
BBOX_THICKNESS = 4          # bbox 线宽
# ================== 配置结束 ===================================

def find_images_recursive(root, exts):
    files = []
    for ext in exts:
        pattern = os.path.join(root, "**", f"*.{ext}")
        files.extend(glob(pattern, recursive=True))
    files = sorted(files)
    return files

def read_label_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    parsed = [line.split() for line in lines]
    return parsed

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def draw_yolo_bboxes_clean(img, labels, color=(0,255,0), thickness=2):
    """
    纯净可视化：仅画边框（无文字、无填充、无透明）
    labels: list of token lists, 每行至少含5个字段 [cls, x, y, w, h, ...]
    """
    h, w = img.shape[:2]
    out = img.copy()
    for parts in labels:
        if len(parts) < 5:
            continue
        # 解析（即便不使用类别，也按格式读取）
        try:
            x_c = float(parts[1]); y_c = float(parts[2])
            bw = float(parts[3]); bh = float(parts[4])
        except Exception:
            continue
        xmin = int((x_c - bw/2) * w); ymin = int((y_c - bh/2) * h)
        xmax = int((x_c + bw/2) * w); ymax = int((y_c + bh/2) * h)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(w-1, xmax); ymax = min(h-1, ymax)
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, thickness)
    return out

def main():
    IMAGES = IMAGES_DIR
    LABELS = LABELS_DIR
    OUT = OUT_DIR
    ensure_dir(OUT)

    # 搜索图片（递归）
    images = find_images_recursive(IMAGES, IMAGE_EXTS)
    if len(images) == 0:
        print(f"[WARN] 没在 {IMAGES} 找到图片（后缀：{IMAGE_EXTS}）。请检查路径或扩展名。")
        return

    # 以图片基名集合作为匹配主依据
    image_basenames = { os.path.splitext(os.path.basename(p))[0] for p in images }

    # 搜索 label 文件（递归）
    label_files = glob(os.path.join(LABELS, "**", "*.txt"), recursive=True)
    label_basenames = { os.path.splitext(os.path.basename(p))[0] for p in label_files }

    # unmatched labels = labels 中有但 images 中没有对应 basenames
    unmatched = sorted(list(label_basenames - image_basenames))
    unmatched_path = os.path.join(OUT, "unmatched_labels.txt")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        for name in unmatched:
            f.write(name + "\n")
    print(f"[INFO] 写入未匹配到图片的 label 名称到: {unmatched_path} （共 {len(unmatched)} 条）")

    # 遍历每张图片进行可视化（只以 images 为主）
    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS, basename + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取图片: {img_path}，跳过。")
            continue

        labels = read_label_file(label_path)

        if not labels and SKIP_NO_LABEL:
            # 跳过保存
            print(f"[SKIP] {basename} 无 label，已跳过（SKIP_NO_LABEL=True）。")
            continue

        if labels:
            vis = draw_yolo_bboxes_clean(img, labels, color=BBOX_COLOR, thickness=BBOX_THICKNESS)
        else:
            # 无 label 时直接保存原图（不写 NO LABEL 文本，保持干净）
            vis = img.copy()

        save_name = basename + ".jpg"
        save_path = os.path.join(OUT, save_name)
        success = cv2.imwrite(save_path, vis)
        if success:
            print(f"[SAVE] {save_path}")
        else:
            print(f"[ERROR] 保存失败: {save_path}")

    print("[DONE] 可视化完成。")

if __name__ == "__main__":
    main()
