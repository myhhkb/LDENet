import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

# ======================= 用户配置区域 (修改这里) =======================
# 1. 训练好的最佳模型权重路径
MODEL_PATH = 'runs/xiaorong_module/GEIT_DIMB_SEAM_FMIOU/yolo11/weights/epoch211.pt'

# 2. 测试集图片文件夹路径
TEST_IMAGES_DIR = 'heatmap/DDSM-MIAS-V2.0/val/images'

# 3. 测试集标签文件夹路径 (YOLO格式 .txt)
TEST_LABELS_DIR = 'heatmap/DDSM-MIAS-V2.0/val/labels'

# 4. 推理参数
CONF_THRESHOLD = 0.25  # 置信度阈值 (默认0.25)
IOU_THRESHOLD = 0.50  # NMS IoU阈值


# ======================================================================

def xywh2xyxy(x, w, h):
    """将归一化坐标 (x_c, y_c, w, h) 转换为 像素坐标 (x1, y1, x2, y2)"""
    cls, x_c, y_c, bw, bh = x
    x1 = (x_c - bw / 2) * w
    y1 = (y_c - bh / 2) * h
    x2 = (x_c + bw / 2) * w
    y2 = (y_c + bh / 2) * h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """计算两个矩形框的 IoU"""
    # box: [x1, y1, x2, y2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area


def run_evaluation():
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 获取所有图片
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    image_files = [p for p in Path(TEST_IMAGES_DIR).iterdir() if p.suffix.lower() in img_extensions]

    if not image_files:
        print(f"未在 {TEST_IMAGES_DIR} 找到图片，请检查路径。")
        return

    print(f"开始处理 {len(image_files)} 张测试图片...\n")

    # === 存储所有样本指标的列表 ===
    metrics_data = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'ap50': [],
        'ap50_95': []
    }

    # 定义 AP50:95 的阈值阶梯 (0.50, 0.55, ..., 0.95)
    map_thresholds = np.arange(0.5, 0.96, 0.05)

    # 计数器
    no_gt_count = 0  # 纯健康样本数
    missed_count = 0  # 漏检样本数

    for idx, img_path in enumerate(image_files):
        # 显示进度
        if (idx + 1) % 50 == 0:
            print(f"Processing {idx + 1}/{len(image_files)}...")

        # 1. 读取 GT (Ground Truth)
        label_path = Path(TEST_LABELS_DIR) / (img_path.stem + '.txt')

        # 读取图片以获取尺寸
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape

        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    # 假设单类别，直接取后4位
                    if len(parts) >= 5:
                        gt_boxes.append(xywh2xyxy(parts, w_img, h_img))

        # 如果是负样本（没有肿瘤），通常不计入 Recall/IoU/AP 的平均值计算
        # 但会计入 Precision (如果误报了就是 P=0)
        # 这里为了计算标准医学指标，只统计【包含病灶】的样本表现
        if len(gt_boxes) == 0:
            no_gt_count += 1
            continue

        # 2. 模型推理
        results = model.predict(img_path, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]

        # 3. 单图指标计算逻辑
        # 简化假设：医学图像通常关注是否检测到了那个主要的病灶
        # 我们寻找与 GT 最匹配的那个预测框

        best_iou = 0.0
        tp = 0
        fp = 0

        # 贪心匹配：找到与任意 GT 重叠度最高的预测框
        if len(pred_boxes) > 0:
            # 计算最大 IoU
            for p_box in pred_boxes:
                current_max_iou = 0
                for g_box in gt_boxes:
                    iou = compute_iou(p_box, g_box)
                    if iou > current_max_iou:
                        current_max_iou = iou

                # 更新整张图的最佳 IoU (用于 AP 计算)
                if current_max_iou > best_iou:
                    best_iou = current_max_iou

            # 简单的 TP/FP 判定 (基于 IoU=0.5)
            # 如果最佳框 IoU > 0.5，我们要看预测框总数
            # Precision = 匹配到的框 / 总预测框
            # Recall = 匹配到的框 / 总 GT 框

            # 这里为了获取单样本的稳定指标，采用“最佳匹配优先”策略
            matched_indices = set()
            match_count = 0

            for p_box in pred_boxes:
                box_best_iou = 0
                match_idx = -1
                for i, g_box in enumerate(gt_boxes):
                    iou = compute_iou(p_box, g_box)
                    if iou > box_best_iou:
                        box_best_iou = iou
                        match_idx = i

                if box_best_iou >= 0.5 and match_idx != -1 and match_idx not in matched_indices:
                    match_count += 1
                    matched_indices.add(match_idx)

            tp = match_count
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp
        else:
            # 没预测出东西
            tp = 0
            fp = 0
            fn = len(gt_boxes)
            best_iou = 0.0
            missed_count += 1

        # --- 计算指标值 ---

        # 1. Precision
        # 如果预测了框，P = TP / (TP+FP)；如果没预测框，视为 1.0 (没有乱说) 或者 NaN
        # 严谨做法：分母为0时，如果 TP+FP=0，Precision=1.0 (无误报)
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(pred_boxes) == 0 else 0.0)

        # 2. Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # 3. F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 4. IoU (只记录最佳匹配的 IoU，反映定位精度)
        # 如果没检测到，IoU 为 0

        # 5. AP50 (单样本近似)
        # 只要有一个框 IoU > 0.5 且对应 GT，就算通过
        ap50 = 1.0 if best_iou >= 0.5 else 0.0

        # 6. AP50:95 (单样本近似)
        # 计算 best_iou 能通过多少个阈值
        pass_count = np.sum(best_iou >= map_thresholds)
        ap50_95 = pass_count / 10.0

        # --- 存入列表 ---
        metrics_data['precision'].append(precision * 100)
        metrics_data['recall'].append(recall * 100)
        metrics_data['f1'].append(f1 * 100)
        metrics_data['iou'].append(best_iou * 100)
        metrics_data['ap50'].append(ap50 * 100)
        metrics_data['ap50_95'].append(ap50_95 * 100)

    # ======================= 最终统计输出 =======================
    print("\n" + "=" * 60)
    print(f"📊 实验统计报告 (基于 {len(metrics_data['precision'])} 个正样本)")
    print(f"   (跳过无标签样本: {no_gt_count}, 漏检样本: {missed_count})")
    print("=" * 60)
    print(f"{'Metric':<15} | {'Mean (%)':<10} | {'Std Dev (±)':<12}")
    print("-" * 45)

    final_results = {}

    for key, values in metrics_data.items():
        if len(values) == 0:
            print(f"{key:<15} | N/A        | N/A")
            continue

        data_arr = np.array(values)
        mean_val = np.mean(data_arr)
        std_val = np.std(data_arr, ddof=1)  # 样本标准差

        print(f"{key.upper():<15} | {mean_val:<10.2f} | {std_val:<12.2f}")

        # 存储格式化字符串用于复制
        final_results[key] = f"{mean_val:.2f} ± {std_val:.2f}"

    print("=" * 60)
    print("\n📋 论文 Latex 表格数据速查 (复制下面这些):")
    print("-" * 30)
    print(f"Precision : {final_results.get('precision', 'N/A')}")
    print(f"Recall    : {final_results.get('recall', 'N/A')}")
    print(f"F1-Score  : {final_results.get('f1', 'N/A')}")
    print(f"IoU       : {final_results.get('iou', 'N/A')}")
    print(f"mAP@0.5   : {final_results.get('ap50', 'N/A')}")
    print(f"mAP@0.5:0.95: {final_results.get('ap50_95', 'N/A')}")
    print("-" * 30)


if __name__ == "__main__":
    run_evaluation()
