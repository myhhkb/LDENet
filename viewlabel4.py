import cv2
import numpy as np
import os

# ============================
#          配置输入
# ============================

image_path = "heatmap/output/b/images/20587758.jpg"          # 输入：图像路径
label_path = "heatmap/output/b/labels/20587758.txt"          # 输入：YOLO标注路径
output_path = "heatmap/output/b/cropped_output/20587758.jpg"     # 输出：结果图像保存路径

padding_ratio = 1# 外扩比例，可调，例如 0.1 表示扩大 10%

# ============================
#          函数部分
# ============================

def yolo_to_xyxy(box, img_w, img_h):
    """ YOLO → 像素坐标(x1,y1,x2,y2) """
    cls, xc, yc, w, h = box
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return cls, x1, y1, x2, y2


def expand_to_square(x1, y1, x2, y2, img_w, img_h, padding_ratio):
    """ 扩展外接矩形，并转为正方形 """

    # 原矩形尺寸
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 扩展（padding 比例）
    long_side = max(w, h)
    long_side = int(long_side * (1 + padding_ratio))

    # 正方形
    half = long_side // 2

    new_x1 = int(cx - half)
    new_y1 = int(cy - half)
    new_x2 = int(cx + half)
    new_y2 = int(cy + half)

    # 边界裁剪
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w - 1, new_x2)
    new_y2 = min(img_h - 1, new_y2)

    return new_x1, new_y1, new_x2, new_y2


# ============================
#          主流程
# ============================

# load image
img = cv2.imread(image_path)
if img is None:
    raise ValueError("❌ 图像加载失败，请检查 image_path")

h, w = img.shape[:2]

# load label
true_boxes = []
with open(label_path, "r") as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.strip().split())
        if int(cls) == 0:   # 只保留真实框
            true_boxes.append((cls, xc, yc, bw, bh))

if len(true_boxes) == 0:
    raise ValueError("❌ 标注文件没有类别 0 的真实框！")

# 所有真实框 → 外接矩形
xs = []
ys = []
for box in true_boxes:
    _, x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
    xs += [x1, x2]
    ys += [y1, y2]

min_x = min(xs)
min_y = min(ys)
max_x = max(xs)
max_y = max(ys)

# 转为正方形外接矩形（加padding）
crop_x1, crop_y1, crop_x2, crop_y2 = expand_to_square(
    min_x, min_y, max_x, max_y, w, h, padding_ratio
)

# 裁剪
crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

# 放大到原图大小
resized = cv2.resize(crop, (w, h))

# 保存新图像
cv2.imwrite(output_path, resized)

print("✅ 已保存裁剪+放大后的图像到：", output_path)
