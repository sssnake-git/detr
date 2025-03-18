# encoding= utf-8

import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import requests
import numpy as np

# COCO 类别标签
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def parse_args():
    parser = argparse.ArgumentParser(description = 'DETR Single Image Inference')
    parser.add_argument('--image', type = str, required = True, help = 'Path to input image')
    parser.add_argument('--threshold', type = float, default = 0.7, help = 'Confidence threshold for detection (default: 0.7)')
    parser.add_argument('--model', required = True, type = str, help = 'DETR model directory')
    parser.add_argument('--device', type = str, choices = ['cpu', 'cuda'], default = 'cpu', help = 'Device to run inference (default: cpu)')
    parser.add_argument('--output', type = str, required = True, help = 'Path to save the output image (optional)')
    return parser.parse_args()

def transform_image(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(800),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def load_model(model_name, device):
    print(f'Loading local model from {model_name}...')
    checkpoint = torch.load(model_name, map_location = device)
    
    # 自动推断模型架构
    model_type = 'detr_resnet50'
    model = torch.hub.load('facebookresearch/detr', model_type, pretrained = False)
    model.load_state_dict(checkpoint['model'])
    
    model.to(device).eval()
    return model

def post_process(outputs, image_size, threshold):
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    bboxes = outputs['pred_boxes'][0, keep]
    scores = probas[keep].max(-1).values
    labels = probas[keep].argmax(-1)

    w, h = image_size
    bboxes = rescale_bboxes(bboxes, (w, h))

    return bboxes, scores, labels

def rescale_bboxes(bboxes, size):
    w, h = size
    bboxes = bboxes * torch.tensor([w, h, w, h], dtype = torch.float32)
    bboxes[:, :2] -= bboxes[:, 2:] / 2
    bboxes[:, 2:] += bboxes[:, :2]
    return bboxes.numpy()

def draw_detections(image, bboxes, scores, labels):
    draw = ImageDraw.Draw(image)
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox
        label_text = f'{CLASSES[label]}: {score:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline = 'red', width = 3)
        draw.text((x1, y1), label_text, fill = 'red')
    return image

def run_inference(args):
    image = Image.open(args.image).convert('RGB')
    transformed_image = transform_image(image)

    device = torch.device(args.device)
    model = load_model(args.model, device)

    transformed_image = transformed_image.to(device)
    with torch.no_grad():
        outputs = model(transformed_image)

    bboxes, scores, labels = post_process(outputs, image.size, args.threshold)
    result_image = draw_detections(image, bboxes, scores, labels)

    result_image.save(args.output)
    print(f'Detection result saved to {args.output}')

if __name__ == '__main__':
    args = parse_args()
    run_inference(args)
