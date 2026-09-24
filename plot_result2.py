import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D

# 1. 完整数据录入 (严格对应 Table 1 数据)
full_data = [
    # YOLO-based Detectors
    ["Gold-YOLO-N", 5.6, 78.9, "YOLO"], ["YOLOv8-N", 3.2, 80.8, "YOLO"],
    ["YOLOv9-T", 2.0, 79.6, "YOLO"], ["YOLOv10-N", 2.3, 78.1, "YOLO"],
    ["YOLOv11-N", 2.6, 81.5, "YOLO"], ["Hyper-YOLO-N", 4.0, 82.0, "YOLO"],
    ["Mamba YOLO-T", 5.8, 79.5, "YOLO"], ["SpikeYOLO", 13.0, 79.3, "YOLO"],
    ["YOLOv12-N", 2.6, 82.2, "YOLO"], ["YOLOv13-N", 2.5, 79.4, "YOLO"],
    # DETR-based Detectors
    ["RT-DETRv2-R18", 20.0, 81.6, "DETR"],
    ["D-FINE-N", 4.0, 79.6, "DETR"], ["DEIM-N", 4.0, 82.4, "DETR"],
    ["DEIMv2-N", 3.6, 82.9, "DETR"],
    # Medical Image Detectors
    ["SSD", 24.5, 75.6, "Medical"], ["RetinaNet", 36.4, 72.1, "Medical"],
    ["CenterNet", 25.0, 80.6, "Medical"], ["ERetinaNet", 37.0, 75.1, "Medical"],
    ["MBMDNet", 44.8, 77.7, "Medical"], ["Gelan", 25.0, 83.7, "Medical"],
    ["AE-YOLO", 8.0, 84.9, "Medical"],
    # Ours (LEINet: 2.6M, 85.8%)
    ["LEINet (Ours)", 2.6, 85.8, "Ours"]
]

# 2. 学术绘图配置
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# 颜色配置
color_map = {"YOLO": "#E1F0F7", "DETR": "#E8E1F0", "Medical": "#FAD7D7", "Ours": "#1f77b4"}

# 3. 气泡绘制逻辑
base_visual_area = 10000
max_p = max([d[1] for d in full_data])

texts = []
scatters = [] # 用于存储散点对象

for name, p, ap, cat in full_data:
    color = color_map[cat]
    is_ours = (cat == "Ours")
    z = 100 if is_ours else 10

    # 气泡大小逻辑
    s = max(180, (p / max_p) * base_visual_area)

    # 绘制气泡并将返回的对象存入列表
    scat = ax.scatter(p, ap, s=s, c=color, alpha=0.85,
                      edgecolors='white', linewidth=0.6, zorder=z)
    scatters.append(scat)

    # 字体配置
    f_size = 8 if is_ours else 6
    f_weight = 'bold' if is_ours else 'normal'

    # 初始化文字位置：稍微向上偏移一点，给算法一个初始方向
    t = ax.text(p, ap + 0.2, name, fontsize=f_size, fontweight=f_weight, zorder=z + 1)
    texts.append(t)

# 4. 坐标轴与范围设置
ax.set_xscale('log')
ax.set_xlabel('Params (M)', fontsize=13, labelpad=8)
ax.set_ylabel('Average Precision (AP$_{50}$ %)', fontsize=13, labelpad=8)
ax.set_xticks([2, 5, 10, 20, 40, 60])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_ylim(70, 87) #

# 5. 改进的自动标签避让
# 通过 add_objects=scatters 让算法强力避开气泡
adjust_text(texts,
            add_objects=scatters,      # 关键修改：避开气泡物体本身
            only_move={'points': 'y', 'text': 'xy'},
            expand_text=(1.5, 1.5),    # 文字间的互斥
            expand_objects=(1.6, 1.6), # 关键修改：文字与气泡边缘的排斥倍数
            force_text=0.75,           # 增加文字间的排斥力
            force_objects=1.0,         # 增加气泡对文字的排斥力
            arrowprops=None)

# 6. 图层修饰
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='both', linestyle=':', alpha=0.3, zorder=0)

legend_elements = [Line2D([0], [0], marker='o', color='w', label=k,
                          markerfacecolor=v, markersize=10) for k, v in color_map.items()]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10) #

plt.tight_layout()
plt.savefig('model_comparison_v3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()