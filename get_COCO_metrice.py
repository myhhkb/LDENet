# import warnings
# warnings.filterwarnings('ignore')
# import argparse
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from tidecv import TIDE, datasets
#
# # COCO指标如果一直生成不出来之类的问题可以看这期视频排查：https://www.bilibili.com/video/BV1SdNizEE4X/
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--anno_json', type=str, default='dataset/data.json', help='label coco json path')
#     parser.add_argument('--pred_json', type=str, default='D:/cs_work/ultralytics-yolo11-main/runs/val/xiaorong/GEIT_DIMB_SEAM_FMIOU/predictions.json', help='pred coco json path')
#
#     return parser.parse_known_args()[0]
#
# if __name__ == '__main__':
#     opt = parse_opt()
#     anno_json = opt.anno_json
#     pred_json = opt.pred_json
#
#     anno = COCO(anno_json)  # init annotations api
#     pred = anno.loadRes(pred_json)  # init predictions api
#     eval = COCOeval(anno, pred, 'bbox')
#     eval.evaluate()
#     eval.accumulate()
#     eval.summarize()
#
#     tide = TIDE()
#     tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
#     tide.summarize()
#     tide.plot(out_dir='result_coco')


import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import csv
import os
import io
from contextlib import redirect_stdout

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='dataset/data_inbreast_640.json', help='label coco json path')
    parser.add_argument('--pred_json', type=str, default='D:/cs_work/Gold-YOLO/tools/runs/val/exp/predictions.json', help='pred coco json path')
    parser.add_argument('--out_dir', type=str, default='result_coco/inbreast_640/Gold-YOLO', help='输出文件夹')
    parser.add_argument('--save_evalImgs', action='store_true', help='是否保存 evalImgs（逐图详细结果，可能很大）')
    return parser.parse_known_args()[0]

# COCO summary 指标的可读名称（12 个 summary）
COCO_SUMMARY_KEYS = [
    "AP", "AP@0.50", "AP@0.75", "AP_small", "AP_medium", "AP_large",
    "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"
]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    out_dir = opt.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 初始化 COCO / 预测 / 评估
    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    eval = COCOeval(anno, pred, 'bbox')

    # 捕获 COCOeval 的 summarize 输出到字符串（便于写入文本日志）
    buf = io.StringIO()
    with redirect_stdout(buf):
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
    cocoeval_text = buf.getvalue()

    # 将 summary 数值写成 JSON / CSV（eval.stats 是一个 numpy 数组或 list）
    stats = eval.stats.tolist() if hasattr(eval.stats, 'tolist') else list(eval.stats)
    summary_dict = {}
    for i, key in enumerate(COCO_SUMMARY_KEYS):
        # 有的实现返回的 stats 数量可能不同，这里防护一下
        summary_dict[key] = float(stats[i]) if i < len(stats) else None

    # 保存 JSON
    with open(os.path.join(out_dir, 'cocoeval_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)

    # 保存 CSV（两列：metric,value）
    with open(os.path.join(out_dir, 'cocoeval_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in summary_dict.items():
            writer.writerow([k, v])

    # 保存人可读的文本（包含原始 summarize 输出）
    with open(os.path.join(out_dir, 'cocoeval_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("===== COCOeval summarize() 输出 =====\n")
        f.write(cocoeval_text)
        f.write("\n\n===== 解析后的 summary JSON =====\n")
        f.write(json.dumps(summary_dict, indent=2, ensure_ascii=False))

    # 如果需要，可以把 evalImgs（逐图细节）保存为 json（注意可能非常大）
    if opt.save_evalImgs:
        # eval.evalImgs 是 list of dict 或 numpy array，直接保存（要小心体积）
        try:
            import numpy as np
            # 将可能含 numpy 类型的结构转为可序列化
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, bytes):
                    return obj.decode()
                return obj
            evalImgs = eval.evalImgs
            # 如果是 dict 或 list，直接 dump
            with open(os.path.join(out_dir, 'cocoeval_evalImgs.json'), 'w', encoding='utf-8') as f:
                json.dump(evalImgs, f, default=make_serializable, indent=2, ensure_ascii=False)
        except Exception as e:
            with open(os.path.join(out_dir, 'cocoeval_evalImgs_error.txt'), 'w', encoding='utf-8') as f:
                f.write("保存 evalImgs 失败: " + str(e))

    # --------------- TIDE 评估 ----------------
    tide = TIDE()
    # 捕获 tide 的打印输出
    tide_buf = io.StringIO()
    with redirect_stdout(tide_buf):
        tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
        tide.summarize()
        # tide.plot 会把图片保存到 out_dir，保持原样
        tide.plot(out_dir=out_dir)
    tide_text = tide_buf.getvalue()

    # 保存 TIDE 文本输出
    with open(os.path.join(out_dir, 'tide_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(tide_text)

    # 最后打印提示（仍然会输出到控制台）
    print(f"已将 COCOeval 指标保存为: {os.path.join(out_dir, 'cocoeval_summary.json')}, .csv, .txt")
    if opt.save_evalImgs:
        print(f"（注意）evalImgs 已保存到 {os.path.join(out_dir, 'cocoeval_evalImgs.json')}")
    print(f"TIDE 输出文本已保存为: {os.path.join(out_dir, 'tide_summary.txt')}")
    print(f"TIDE 绘图文件保存在: {out_dir} （tide.plot 生成）")
