#!/usr/bin/env python3
# compute_iou_from_label.py
# 用法1（在脚本中直接设置路径）:
#   修改下方 LABEL_PATH 然后运行 python3 compute_iou_from_label.py
# 用法2（命令行传参）:
#   python3 compute_iou_from_label.py /path/to/label.txt

import os
import sys

LABEL_PATH = 'heatmap/DDSM-MIAS-V2.0/yc/7/train/labels/B_3504_1_RIGHT_CC_jpg.rf.d8869c9f07d727e43cc930d04f833608.txt'  # 如果想直接在脚本中指定路径，把路径写在这里，如 '/path/to/label.txt'
OUTPUT_TXT = 'heatmap/DDSM-MIAS-V2.0/yc/iou_results.txt'  # 结果追加到当前目录的这个文件里

def read_yolo_label_file(path):
    """读取YOLO标注文件，返回 [(class, [cx,cy,w,h]), ...]"""
    boxes = []
    if not os.path.exists(path):
        raise FileNotFoundError(f'Label file not found: {path}')
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:5]))
                boxes.append((cls, coords))
            except Exception:
                continue
    return boxes

def yolo_to_xyxy_normalized(box):
    """输入 [cx,cy,w,h]（归一化），返回 [x1,y1,x2,y2]（归一化）"""
    cx, cy, w, h = box
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [x1, y1, x2, y2]

def iou_norm(boxA, boxB):
    """输入两个归一化的xyxy框，返回IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, boxA[2]-boxA[0]) * max(0.0, boxA[3]-boxA[1])
    areaB = max(0.0, boxB[2]-boxB[0]) * max(0.0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

def greedy_match_iou_matrix(iou_matrix):
    """
    贪心匹配：每次选择当前最大的 IoU (i,j)，分配 GT i - Pred j，然后移除对应行列。
    返回 matched_pairs: list of (gt_idx, pred_idx, iou)
    """
    import math
    matched = []
    if iou_matrix.size == 0:
        return matched
    # copy
    mat = iou_matrix.copy()
    # we'll mark assigned rows/cols by setting to -1
    while True:
        # find max
        max_val = mat.max()
        if max_val <= 0:
            break
        # get its indices (first occurrence)
        # flatten search for indices
        idx = None
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] == max_val:
                    idx = (i, j)
                    break
            if idx is not None:
                break
        if idx is None:
            break
        i, j = idx
        matched.append((i, j, float(max_val)))
        # set row i and col j to -1 so they won't be chosen again
        mat[i, :] = -1.0
        mat[:, j] = -1.0
    return matched

def compute_and_record_iou(label_path, output_txt=OUTPUT_TXT, verbose=True):
    boxes = read_yolo_label_file(label_path)
    gts = [b[1] for b in boxes if int(b[0]) == 0]
    preds = [b[1] for b in boxes if int(b[0]) == 1]

    # convert to xyxy normalized
    gts_xy = [yolo_to_xyxy_normalized(x) for x in gts]
    preds_xy = [yolo_to_xyxy_normalized(x) for x in preds]

    # if no predictions or no gt, handle separately
    if len(preds_xy) == 0:
        iou_list = []
    elif len(gts_xy) == 0:
        # no GT, all preds -> IoU = 0
        iou_list = [0.0] * len(preds_xy)
    else:
        # compute IoU matrix (gt x pred)
        import numpy as np
        iou_mat = np.zeros((len(gts_xy), len(preds_xy)), dtype=float)
        for i in range(len(gts_xy)):
            for j in range(len(preds_xy)):
                iou_mat[i, j] = iou_norm(gts_xy[i], preds_xy[j])
        # greedy match
        matched = greedy_match_iou_matrix(iou_mat)
        # build pred_idx -> matched_iou mapping
        pred_to_iou = { j: 0.0 for j in range(len(preds_xy)) }  # default 0
        for gt_i, pred_j, iouv in matched:
            pred_to_iou[pred_j] = iouv
        # Now we must return IoUs in left-to-right order of predictions.
        # Determine center x of each pred for sorting
        pred_centers_x = [p[0] for p in preds]  # original cx from yolo
        # create list of (pred_idx, center_x) then sort by center_x ascending
        idx_order = sorted(list(range(len(preds_xy))), key=lambda k: pred_centers_x[k])
        iou_list = [pred_to_iou[idx] for idx in idx_order]

    # format output string "IOU = v1,v2,..."
    iou_str = ','.join([f'{v:.6f}'.rstrip('0').rstrip('.') if v!=0 else '0.0' for v in iou_list])
    output_line = f'IOU = {iou_str}' if iou_str != '' else 'IOU = '

    # 分类记录（根据标注文件名字分类）
    base = os.path.basename(label_path)
    name_noext = os.path.splitext(base)[0]
    # 这里把类别标识为文件名前缀（如 user 要更细的分类可根据需要修改）
    category = name_noext

    # write (append) to the output txt
    with open(output_txt, 'a', encoding='utf-8') as fout:
        fout.write(f'{category}\t{output_line}\n')

    if verbose:
        print(output_line)
        print(f'Result appended to {output_txt} as category "{category}"')

    return iou_list

if __name__ == '__main__':
    # 支持命令行传入路径
    label_path = LABEL_PATH
    if len(sys.argv) > 1:
        label_path = sys.argv[1]
    if label_path is None:
        print('请在脚本中设置 LABEL_PATH 或者通过命令行传入标签文件路径')
        print('示例: python3 compute_iou_from_label.py /path/to/label.txt')
        sys.exit(1)

    compute_and_record_iou(label_path)
