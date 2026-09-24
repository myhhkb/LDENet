import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图像: {path}')
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else '.jpg'
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f'无法编码图像: {path}')
    encoded.tofile(str(path))


def parse_yolo_bbox(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        values = [float(x) for x in parts[1:5]]
        cx, cy, bw, bh = values
        x1 = int(round((cx - bw / 2) * width))
        y1 = int(round((cy - bh / 2) * height))
        x2 = int(round((cx + bw / 2) * width))
        y2 = int(round((cy + bh / 2) * height))
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def parse_yolo_segments(mask_path, width, height):
    polygons = []
    if not mask_path.exists():
        return polygons

    for line in mask_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        points = []
        for x, y in zip(coords[0::2], coords[1::2]):
            px = int(round(x * width))
            py = int(round(y * height))
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            points.append([px, py])
        if len(points) >= 3:
            polygons.append(np.array(points, dtype=np.int32))
    return polygons


def gaussian_from_mask(mask, sigma=11.0, gamma=0.85):
    mask = mask.astype(np.float32) / 255.0
    if sigma > 0:
        ksize = int(2 * round(3 * sigma) + 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    max_value = float(mask.max())
    if max_value > 1e-6:
        mask = mask / max_value
    if gamma != 1.0:
        mask = np.power(mask, gamma)
    return mask.astype(np.float32)


def bbox_gaussian_heatmap(shape, boxes, circular=True, radius_scale=0.50):
    height, width = shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        if circular:
            sigma = max(min(bw, bh) * radius_scale, 1.0)
            dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
            current = np.exp(-dist2 / (2.0 * sigma ** 2))
        else:
            sigma_x = max(bw * radius_scale, 1.0)
            sigma_y = max(bh * radius_scale, 1.0)
            dist2 = ((xs - cx) / sigma_x) ** 2 + ((ys - cy) / sigma_y) ** 2
            current = np.exp(-dist2 / 2.0)
        heatmap = np.maximum(heatmap, current.astype(np.float32))

    max_value = float(heatmap.max())
    if max_value > 1e-6:
        heatmap = heatmap / max_value
    return heatmap


def build_annotation_heatmap(image_shape, mask_path, label_path, use_mask=True,
                             mask_sigma=11.0, mask_gamma=0.85,
                             circular_bbox=True, bbox_radius_scale=0.50):
    height, width = image_shape[:2]

    if use_mask:
        polygons = parse_yolo_segments(mask_path, width, height)
        if polygons:
            binary = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(binary, polygons, 255)
            return gaussian_from_mask(binary, sigma=mask_sigma, gamma=mask_gamma), 'mask'

    boxes = parse_yolo_bbox(label_path, width, height)
    if boxes:
        return bbox_gaussian_heatmap(
            (height, width),
            boxes,
            circular=circular_bbox,
            radius_scale=bbox_radius_scale,
        ), 'bbox'

    return np.zeros((height, width), dtype=np.float32), 'empty'


def overlay_heatmap(image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET,
                    threshold=0.03, keep_background=True):
    heatmap = np.clip(heatmap, 0.0, 1.0)
    visible = heatmap >= threshold
    heatmap_u8 = np.uint8(heatmap * 255)
    color_map = cv2.applyColorMap(heatmap_u8, colormap)

    if keep_background:
        output = image.copy()
        blended = cv2.addWeighted(image, 1.0 - alpha, color_map, alpha, 0)
        output[visible] = blended[visible]
        return output

    output = np.zeros_like(image)
    output[visible] = color_map[visible]
    return output


def process_one(image_path, labels_dir, masks_dir, output_dir, args):
    image = read_image(image_path)
    stem = image_path.stem
    label_path = labels_dir / f'{stem}.txt'
    mask_path = masks_dir / f'{stem}.txt'

    heatmap, source = build_annotation_heatmap(
        image.shape,
        mask_path,
        label_path,
        use_mask=not args.no_mask,
        mask_sigma=args.mask_sigma,
        mask_gamma=args.mask_gamma,
        circular_bbox=not args.ellipse_bbox,
        bbox_radius_scale=args.bbox_radius_scale,
    )
    output = overlay_heatmap(
        image,
        heatmap,
        alpha=args.alpha,
        threshold=args.threshold,
        keep_background=not args.only_heatmap,
    )

    save_path = output_dir / image_path.name
    save_image(save_path, output)
    return source, save_path


def parse_args():
    parser = argparse.ArgumentParser(description='根据 YOLO bbox/segmentation 标注生成热力图')
    parser.add_argument('--input', default='heatmap/input', help='输入根目录, 包含 images/ labels/ mask/')
    parser.add_argument('--output', default='heatmap/output/annotation_heatmap/images', help='输出图像目录')
    parser.add_argument('--alpha', type=float, default=0.45, help='热力图叠加透明度')
    parser.add_argument('--threshold', type=float, default=0.03, help='低于该值的热力图不显示')
    parser.add_argument('--mask-sigma', type=float, default=11.0, help='分割 mask 平滑强度')
    parser.add_argument('--mask-gamma', type=float, default=0.85, help='分割热力图亮度曲线, 小于 1 会扩大高亮范围')
    parser.add_argument('--bbox-radius-scale', type=float, default=0.50, help='无 mask 时 bbox 高斯半径比例')
    parser.add_argument('--ellipse-bbox', action='store_true', help='无 mask 时使用椭圆高斯, 默认使用圆形高斯')
    parser.add_argument('--no-mask', action='store_true', help='忽略 mask, 只根据 labels bbox 生成')
    parser.add_argument('--only-heatmap', action='store_true', help='只输出热力图颜色, 不叠加原图背景')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'labels'
    masks_dir = input_dir / 'mask'
    output_dir = Path(args.output)

    if not images_dir.exists():
        raise FileNotFoundError(f'图像目录不存在: {images_dir}')

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f'没有找到图像文件: {images_dir}')

    count_by_source = {'mask': 0, 'bbox': 0, 'empty': 0}
    for image_path in image_paths:
        source, save_path = process_one(image_path, labels_dir, masks_dir, output_dir, args)
        count_by_source[source] += 1
        print(f'[{source}] {image_path.name} -> {save_path}')

    print('完成:', count_by_source)


if __name__ == '__main__':
    main()
