#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_two_classes_01.py

- YOLO 归一化标签（class x_center y_center width height ...）
- 只画 bbox（无文字、无填充）
- class 0 -> 绿色 (RGB: 0,255,0)
- class 1 -> 红色   (RGB: 255,0,0)
- 其他类被忽略（不画）
- 以 images 目录为主遍历，labels 多出的基名写入 unmatched_labels.txt
- 直接在顶部配置参数（不使用命令行）
"""

import os
from glob import glob
import cv2

# ================== 在这里配置（修改这部分即可，使用 RGB 格式） ==================
IMAGES_DIR = "heatmap/DDSM-MIAS-V2.0/yc/7/train/images"     # 图片根目录（会递归查找）
LABELS_DIR = "heatmap/DDSM-MIAS-V2.0/yc/7/train/labels"     # labels 目录（按 basename + .txt 匹配）
OUT_DIR = "heatmap/DDSM-MIAS-V2.0/yc/VIS2"       # 输出目录（自动创建）
IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp"]   # 要搜索的图片后缀（小写）
SKIP_NO_LABEL = False                        # True 则跳过没有 label 的图片（不保存）
BBOX_THICKNESS = 1                           # 框线宽

# 颜色配置（使用直观的 RGB 值）
# COLOR_CLASS_0_RGB = (0, 255, 0)   # class 0 -> 绿色 (R,G,B)
COLOR_CLASS_1_RGB = (0, 0, 255)   # class 0 -> 蓝色 (R,G,B)
COLOR_CLASS_0_RGB = (255, 0, 0)   # class 1 -> 红色

# ================== 配置结束 ===================================

def rgb_to_bgr(rgb):
    r, g, b = rgb
    return (int(b), int(g), int(r))

COLOR_CLASS_0 = rgb_to_bgr(COLOR_CLASS_0_RGB)
COLOR_CLASS_1 = rgb_to_bgr(COLOR_CLASS_1_RGB)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_images_recursive(root, exts):
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
    files = sorted(files)
    return files

def read_label_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return [ln.split() for ln in lines]

def draw_bboxes_0_1(img, labels, thickness=2):
    """
    仅绘制 class 0 与 class 1 的 bbox。
    labels: list of token lists, 每行至少包含 [cls, x, y, w, h]
    """
    h, w = img.shape[:2]
    out = img.copy()
    for parts in labels:
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x_c = float(parts[1]); y_c = float(parts[2])
            bw = float(parts[3]); bh = float(parts[4])
        except Exception:
            continue

        # 只处理 0 和 1
        if cls not in (0, 1):
            continue

        xmin = int((x_c - bw/2) * w)
        ymin = int((y_c - bh/2) * h)
        xmax = int((x_c + bw/2) * w)
        ymax = int((y_c + bh/2) * h)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(w-1, xmax); ymax = min(h-1, ymax)

        color = COLOR_CLASS_0 if cls == 0 else COLOR_CLASS_1
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, thickness)

    return out

def main():
    IMAGES = IMAGES_DIR
    LABELS = LABELS_DIR
    OUT = OUT_DIR
    ensure_dir(OUT)

    # 查找图片（递归）
    images = find_images_recursive(IMAGES, IMAGE_EXTS)
    if not images:
        print(f"[WARN] 在 {IMAGES} 未找到图片（后缀: {IMAGE_EXTS}）。请检查路径或后缀。")
        return

    image_basenames = { os.path.splitext(os.path.basename(p))[0] for p in images }

    # 查找所有 label 文件（递归）
    label_files = glob(os.path.join(LABELS, "**", "*.txt"), recursive=True)
    label_basenames = { os.path.splitext(os.path.basename(p))[0] for p in label_files }

    # unmatched: label 有但 images 没有对应基名
    unmatched = sorted(list(label_basenames - image_basenames))
    unmatched_path = os.path.join(OUT, "unmatched_labels.txt")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        for name in unmatched:
            f.write(name + "\n")
    print(f"[INFO] unmatched labels 写入: {unmatched_path} （共 {len(unmatched)} 条）")

    # 对 images 中每张图片做可视化
    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS, basename + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 无法读取图片: {img_path}，跳过。")
            continue

        labels = read_label_file(label_path)
        if not labels and SKIP_NO_LABEL:
            print(f"[SKIP] {basename} 无 label，已跳过。")
            continue

        if labels:
            vis = draw_bboxes_0_1(img, labels, thickness=BBOX_THICKNESS)
        else:
            # 无 label 时直接保存原图（不标注任何文字）
            vis = img.copy()

        save_path = os.path.join(OUT, basename + ".jpg")
        ok = cv2.imwrite(save_path, vis)
        if ok:
            print(f"[SAVE] {save_path}")
        else:
            print(f"[ERROR] 保存失败: {save_path}")

    print("[DONE] 可视化完成。")

if __name__ == "__main__":
    main()
