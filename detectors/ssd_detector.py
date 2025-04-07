#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SSD 객체 탐지기 구현 (PyTorch 기반)
torchvision 패키지를 사용하여 SSD 모델로 객체 탐지 수행
"""

import os
import numpy as np
import torch
import logging
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F
from pathlib import Path

logger = logging.getLogger(__name__)

class SSDDetector:
    """SSD 객체 탐지기 클래스"""
    
    def __init__(self, model_path=None, conf_thres=0.5, iou_thres=0.45, device='', classes=None):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes if classes is not None else [1]  # 기본값: 사람 클래스만 탐지 (COCO에서 ID 1)
        
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = f'cuda:{device}' if device else 'cuda:0'
        
        try:
            logger.info("SSD 모델 로드 중...")
            if model_path and os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"사용자 지정 모델 로드 완료: {model_path}")
            else:
                self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
                logger.info("사전 학습된 SSD 모델 로드 완료")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info(f"SSD 모델 로드 완료 (device: {self.device})")
        
        except Exception as e:
            logger.error(f"SSD 모델 로드 실패: {e}")
            raise
    
    def detect(self, image):
        image_rgb = np.copy(image)  # Negative stride 방지
        image_tensor = F.to_tensor(image_rgb).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()  # [x1, y1, x2, y2]
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score < self.conf_thres or (self.classes and label not in self.classes):
                continue
            
            x1, y1, x2, y2 = box.astype(float)
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(label)
            }
            detections.append(detection)
        
        return detections