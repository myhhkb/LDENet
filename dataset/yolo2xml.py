import os
import cv2
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET


def make_xml_entry(filename, width, height, channel, objects):
    """
    构建 XML 的层级结构
    """
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = filename

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(channel)

    ET.SubElement(annotation, 'segmented').text = '0'

    for obj in objects:
        object_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(object_elem, 'name').text = obj['name']
        ET.SubElement(object_elem, 'pose').text = 'Unspecified'
        ET.SubElement(object_elem, 'truncated').text = '0'
        ET.SubElement(object_elem, 'difficult').text = '0'

        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(obj['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(obj['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(obj['ymax'])

    return annotation


def yolo_to_xml(image_dir, txt_dir, xml_dir, classes):
    """
    执行批量转换
    :param image_dir: 图片文件夹路径
    :param txt_dir: YOLO txt 文件夹路径
    :param xml_dir: 输出 XML 文件夹路径
    :param classes: 类别名称列表 (索引必须与 YOLO class_id 对应)
    """
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    # 遍历所有 txt 文件
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        basename = os.path.splitext(txt_file)[0]
        image_name = basename + ".jpg"  # 假设图片格式为 jpg，需根据实际情况修改
        image_path = os.path.join(image_dir, image_name)
        txt_path = os.path.join(txt_dir, txt_file)

        # 读取图片以获取 H, W
        if not os.path.exists(image_path):
            print(f"警告: 找不到图片 {image_path}，跳过该文件。")
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue
        height, width, channel = img.shape

        objects = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) < 5: continue

                class_id = int(data[0])
                x_center, y_center = float(data[1]), float(data[2])
                w, h = float(data[3]), float(data[4])

                # 数学转换 (反归一化)
                w_abs = w * width
                h_abs = h * height
                x_center_abs = x_center * width
                y_center_abs = y_center * height

                # 计算角点并处理边界 (Clamp)
                xmin = int(x_center_abs - (w_abs / 2))
                ymin = int(y_center_abs - (h_abs / 2))
                xmax = int(x_center_abs + (w_abs / 2))
                ymax = int(y_center_abs + (h_abs / 2))

                # 确保坐标不越界
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)

                obj_info = {
                    'name': classes[class_id],  # 将 ID 映射回类别名
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                }
                objects.append(obj_info)

        # 生成 XML
        tree_root = make_xml_entry(image_name, width, height, channel, objects)

        # 格式化并保存
        xml_str = parseString(ET.tostring(tree_root)).toprettyxml(indent="    ")
        output_path = os.path.join(xml_dir, basename + '.xml')
        with open(output_path, 'w') as f:
            f.write(xml_str)

    print(f"转换完成。XML文件已保存至: {xml_dir}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 请在此处定义你的类别列表，顺序必须与 YOLO 训练时的 classes.txt 一致
    CLASS_LIST = ['mass']

    IMG_PATH = '../heatmap/DDSM-MIAS-V2.0/train/images'  # 图片路径
    TXT_PATH = '../heatmap/DDSM-MIAS-V2.0/train/labels'  # YOLO 标签路径
    XML_PATH = '../heatmap/DDSM-MIAS-V2.0/train/Annotations'  # 输出路径

    yolo_to_xml(IMG_PATH, TXT_PATH, XML_PATH, CLASS_LIST)