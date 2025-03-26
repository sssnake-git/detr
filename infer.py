# encoding = utf-8

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection

# === 参数设定 ===
MODEL_PATH = "models/detr_custom.pth"
IMAGE_PATH = "test.jpg"
NUM_CLASSES = 3  # 替换为你的类别数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载训练好的模型 ===
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet50")  # 先初始化
if torch.cuda.is_available():
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# 修改分类层
in_features = model.class_labels_classifier.in_features
model.class_labels_classifier = torch.nn.Linear(in_features, NUM_CLASSES + 1)
model.bbox_predictor = torch.nn.Linear(in_features, NUM_CLASSES * 4)

model.to(DEVICE)
model.eval()

# === 数据预处理 ===
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 读取测试图片 ===
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# === 推理 ===
with torch.no_grad():
    outputs = model(image_tensor)

# === 解析预测结果 ===
scores = outputs.logits.softmax(-1)[0, :, :-1]  # 去掉背景类
boxes = outputs.pred_boxes[0]  # 预测边界框

# === 绘制检测结果 ===
draw = ImageDraw.Draw(image)
threshold = 0.7  # 置信度阈值

for score, (x, y, w, h) in zip(scores, boxes):
    conf, class_id = score.max(0)
    if conf > threshold:
        x_min, y_min, x_max, y_max = x - w/2, y - h/2, x + w/2, y + h/2
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min), f"Class {class_id.item()} {conf:.2f}", fill="red")

# === 显示检测结果 ===
plt.imshow(image)
plt.axis("off")
plt.show()
