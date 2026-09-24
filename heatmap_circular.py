import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import os, shutil
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

from heatmap import yolo_heatmap, letterbox


class SafeYOLODetectTarget(torch.nn.Module):
    """检测任务 Grad-CAM 目标: 即使最高候选低于阈值, 也至少保留 top-1 用于反传."""

    def __init__(self, output_type, conf, ratio, end2end):
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        num_candidates = max(1, int(post_result.size(0) * self.ratio))
        num_candidates = min(num_candidates, int(post_result.size(0)))

        for i in range(num_candidates):
            score = post_result[i, 0] if self.end2end else post_result[i].max()
            if i > 0 and float(score) < self.conf:
                break

            if self.output_type in ['class', 'all']:
                result.append(score)
            if self.output_type in ['box', 'all']:
                for j in range(4):
                    result.append(pre_post_boxes[i, j])

        if not result:
            return post_result[0, 0] if self.end2end else post_result[0].max()
        return torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x, device=post_result.device) for x in result]).sum()


def gaussian_smooth(cam, sigma):
    """对 CAM 做高斯平滑, 让激活块更接近圆形/椭圆形而不是块状."""
    if sigma is None or sigma <= 0:
        return cam
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(cam, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def radial_weight(h, w, center=None, radius_scale=0.42, circular=True):
    """生成圆形/椭圆形高斯先验, 用于把框内热区拉回到目标中心附近."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if center is None:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    else:
        cy, cx = center

    if circular:
        sigma = max(min(h, w) * radius_scale, 1.0)
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
        weight = np.exp(-dist2 / (2.0 * sigma ** 2))
    else:
        sigma_y = max(h * radius_scale, 1.0)
        sigma_x = max(w * radius_scale, 1.0)
        dist2 = ((xs - cx) / sigma_x) ** 2 + ((ys - cy) / sigma_y) ** 2
        weight = np.exp(-dist2 / 2.0)

    return weight.astype(np.float32)


def cam_centroid(roi, fallback_center):
    """根据框内 CAM 强响应估计目标中心, 原 CAM 不可靠时回退到检测框中心."""
    roi = scale_cam_image(roi.astype(np.float32))
    high = roi >= np.percentile(roi, 75)
    weights = roi * high.astype(np.float32)
    total = float(weights.sum())
    if total <= 1e-6:
        return fallback_center

    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float32)
    cy = float((ys * weights).sum() / total)
    cx = float((xs * weights).sum() / total)
    return cy, cx


def constrain_center(center, h, w, max_offset_ratio=0.18):
    """限制 CAM 重心不能偏离框中心太远, 避免被半边高响应带偏."""
    cy, cx = center
    box_cy, box_cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dy = h * max_offset_ratio
    max_dx = w * max_offset_ratio
    cy = float(np.clip(cy, box_cy - max_dy, box_cy + max_dy))
    cx = float(np.clip(cx, box_cx - max_dx, box_cx + max_dx))
    return cy, cx


def circular_region_mask(h, w, center, region_scale=0.58, soft_edge=0.12, circular=True):
    """生成目标周围的可视化范围, 圆/椭圆外热力图置零, 边缘做柔和过渡."""
    cy, cx = center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    if circular:
        radius = max(min(h, w) * region_scale, 1.0)
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        edge = max(radius * soft_edge, 1.0)
    else:
        ry = max(h * region_scale, 1.0)
        rx = max(w * region_scale, 1.0)
        dist = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
        radius = 1.0
        edge = max(soft_edge, 1e-3)

    mask = np.clip((radius + edge - dist) / edge, 0.0, 1.0)
    return mask.astype(np.float32)

class yolo_heatmap_circular(yolo_heatmap):
    """在原始 yolo_heatmap 基础上优化圆形目标的热力图显示:
    1. 对 CAM 做高斯平滑, 减少网格状、块状和柱状纹理;
    2. 在每个检测框内根据强响应估计目标中心, 但限制中心不能过度偏离框中心;
    3. 生成圆形高斯先验并与原 CAM 融合, 把半圆/柱状响应拉成更符合圆形目标的热区;
    4. 使用圆形/椭圆形范围 mask, 只展示目标附近一定范围, 不显示完整矩形框区域;
    5. 框重叠时取最大响应, 保留多目标情况下的可视化稳定性.
    """

    def __init__(self, *args, smooth_sigma=5.0, prior_strength=0.75,
                 radius_scale=0.32, keep_original=0.25, circular=True,
                 center_offset_ratio=0.18, region_scale=0.58,
                 soft_edge=0.12, **kwargs):
        # 原始 target 在没有候选超过 conf_threshold 时可能返回普通整数 0,
        # Grad-CAM 无法对它反向传播; 这里改成稳健版 target。
        if self.task == 'detect':
            self.target = SafeYOLODetectTarget(
                self.backward_type,
                self.conf_threshold,
                self.ratio,
                self.model.end2end,
            )
        # smooth_sigma: 高斯平滑强度, 越大越圆润(建议 3~8)
        self.smooth_sigma = smooth_sigma
        # prior_strength: 圆形先验强度, 越大越接近圆形目标(建议 0.6~0.9)
        self.prior_strength = prior_strength
        # radius_scale: 热区半径比例, 越小热点越集中(建议 0.25~0.45)
        self.radius_scale = radius_scale
        # keep_original: 保留原 CAM 细节比例, 越小越能抑制半圆/柱状(建议 0.15~0.35)
        self.keep_original = keep_original
        # circular=True 强制圆形; False 则允许跟随检测框形成椭圆形
        self.circular = circular
        # 限制 CAM 重心相对检测框中心的最大偏移比例
        self.center_offset_ratio = center_offset_ratio
        # region_scale: 最终可视化区域大小, 越小越只显示目标附近范围
        self.region_scale = region_scale
        # soft_edge: 可视化区域边缘柔和程度, 0 附近为硬边界
        self.soft_edge = soft_edge

    def process(self, img_path, save_path):
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            print(f"Warning... {img_path} read failure.")
            return

        img, _, (top, bottom, left, right) = letterbox(
            img,
            new_shape=(self.img_size, self.img_size),
            auto=True,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        print(f'tensor size:{tensor.size()}')

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except Exception as e:
            print(f"Warning... self.method(tensor, [self.target]) failure: {type(e).__name__}: {e}")
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]
        if self.renormalize and self.task in ['detect', 'segment', 'pose']:
            boxes = pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32)
            if len(boxes) > 0:
                cam_image = self.renormalize_cam_in_bounding_boxes(boxes, img, grayscale_cam)
            else:
                print("Warning... no boxes detected, save original CAM without circular region mask.")
        if self.show_result:
            cam_image = pred.plot(
                img=cam_image,
                conf=True,
                font_size=None,
                line_width=None,
                labels=False,
            )

        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """重写: 框内圆形化处理."""
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        smoothed = gaussian_smooth(grayscale_cam.astype(np.float32), self.smooth_sigma)

        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2 = min(grayscale_cam.shape[1] - 1, int(x2))
            y2 = min(grayscale_cam.shape[0] - 1, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = smoothed[y1:y2, x1:x2].copy()
            roi = scale_cam_image(roi)
            h, w = roi.shape
            fallback_center = ((h - 1) / 2.0, (w - 1) / 2.0)
            center = cam_centroid(roi, fallback_center)
            center = constrain_center(center, h, w, self.center_offset_ratio)

            circular_prior = radial_weight(
                h,
                w,
                center=center,
                radius_scale=self.radius_scale,
                circular=self.circular,
            )
            circular_prior = scale_cam_image(circular_prior)

            roi_enhanced = scale_cam_image(roi * circular_prior)
            blended = (
                self.keep_original * roi
                + self.prior_strength * circular_prior
                + (1.0 - self.keep_original) * roi_enhanced
            )
            blended = scale_cam_image(blended)

            region_mask = circular_region_mask(
                h,
                w,
                center=center,
                region_scale=self.region_scale,
                soft_edge=self.soft_edge,
                circular=self.circular,
            )
            blended = blended * region_mask

            renormalized_cam[y1:y2, x1:x2] = np.maximum(
                renormalized_cam[y1:y2, x1:x2], blended)

        renormalized_cam = scale_cam_image(renormalized_cam)
        cam_image = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return cam_image


def get_params():
    params = {
        'weight': 'runs/YOLO12/yolo12-origin/weights/best.pt',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        # 只用较深、语义更强的层, 减少浅层高分辨率带来的网格状纹理
        'layer': [16, 18],
        'backward_type': 'all',
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_result': False,
        'renormalize': True,   # 必须为 True 才会走圆形化处理
        'task': 'detect',
        'img_size': 640,
        # ---- 圆形化相关参数 ----
        'smooth_sigma': 6.0,       # 高斯平滑强度, 越大越圆润
        'prior_strength': 0.75,    # 圆形先验强度, 越大越像圆形
        'radius_scale': 0.32,      # 圆形热区半径, 越小热点越集中
        'keep_original': 0.20,     # 保留原 CAM 细节比例, 越小越能抑制半圆/柱状
        'circular': True,          # True 强制圆形; False 允许椭圆
        'center_offset_ratio': 0.15, # 限制中心偏移, 防止半边响应把圆心带偏
        'region_scale': 0.58,      # 最终只显示目标附近圆形范围, 越小范围越小
        'soft_edge': 0.12,         # 圆形范围边缘柔和程度, 越小边界越硬
    }
    return params


if __name__ == '__main__':
    model = yolo_heatmap_circular(**get_params())
    model(r'heatmap/DDSM-MIAS-V2.0/val/images/A_1021_1.RIGHT_MLO.jpg',
          'heatmap/output/YOLO12_circular/A_1021_1.RIGHT_MLO')
    # 批量处理整个文件夹:
    # model(r'heatmap/DDSM-MIAS-V2.0/val/images', 'heatmap/output/YOLO12_circular')
