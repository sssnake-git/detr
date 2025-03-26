# encoding = utf-8

import os
import json
import cv2
from sklearn.model_selection import train_test_split

'''
    yolo label format:
        class_id x_center y_center width height
    detr coco format:
        {
        'images': [{'id': 1, 'file_name': 'img1.jpg', 'height': 720, 'width': 1280}],
        'annotations': [{'image_id': 1, 'category_id': 1, 'bbox': [x_min, y_min, width, height], 'area': w*h, 'iscrowd': 0}],
        'categories': [{'id': 1, 'name': 'person'}]
        }
'''

# YOLO 数据集路径
image_dir = '../inside_detect/250121/images'
label_dir = '../inside_detect/250121/labels'
output_dir = '../inside_detect/250121/annotations_detr'
os.makedirs(output_dir, exist_ok = True)

# 类别列表（根据你的数据集调整）
categories = [
    {"id": 0, "name": "abnormal"},
    {"id": 1, "name": "weld_seam"}
]
cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}

# 获取所有图像文件
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

# 划分训练集和验证集（例如 80% 训练，20% 验证）
train_files, val_files = train_test_split(image_files, test_size = 0.2, random_state = 42)

# 转换函数
def convert_yolo_to_coco(image_files, split_name):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    ann_id = 1  # 标注 ID 从 1 开始递增

    for img_idx, img_file in enumerate(image_files):
        print(img_idx)
        # 读取图像尺寸
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue
        height, width = img.shape[:2]

        # 添加图像信息
        coco_data["images"].append({
            "id": img_idx + 1,  # 图像 ID 从 1 开始
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # 检查对应的 YOLO 标注文件是否存在
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                class_id = int(class_id)

                # 转换为像素坐标
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                box_w = w * width
                box_h = h * height

                # 添加标注信息
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_idx + 1,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0
                })
                ann_id += 1
        else:
            print(f"提示：图像 {img_file} 没有对应的标注文件，视为无目标图像")

    # 保存为 JSON 文件
    output_path = os.path.join(output_dir, f"{split_name}.json")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent = 4)
    print(f"已保存 {split_name} 标注到 {output_path}")

# 生成训练集和验证集的 COCO 标注
convert_yolo_to_coco(train_files, "train")
convert_yolo_to_coco(val_files, "val")

# 调整数据集结构以适配 DETR
os.makedirs("../inside_detect/250121/detr/train", exist_ok = True)
os.makedirs("../inside_detect/250121/detr/val", exist_ok = True)
os.makedirs("../inside_detect/250121/detr/annotations", exist_ok = True)

import shutil
for f in train_files:
    shutil.copy(os.path.join(image_dir, f), "../inside_detect/250121/detr/train")
for f in val_files:
    shutil.copy(os.path.join(image_dir, f), "../inside_detect/250121/detr/val")

shutil.move(os.path.join(output_dir, "train.json"), 
        "../inside_detect/250121/detr/annotations/train.json")
shutil.move(os.path.join(output_dir, "val.json"), 
        "../inside_detect/250121/detr/annotations/annotations/val.json")
print("数据集结构调整完成！")
