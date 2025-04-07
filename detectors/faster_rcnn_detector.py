#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Faster R-CNN 객체 탐지기 구현 (PyTorch 기반)
torchvision 패키지를 사용하여 Faster R-CNN 모델로 객체 탐지 수행
"""

import os
import numpy as np
import torch
import logging
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from pathlib import Path

logger = logging.getLogger(__name__)

class FasterRCNNDetector:
    """Faster R-CNN 객체 탐지기 클래스"""
    
    def __init__(self, model_path=None, conf_thres=0.5, iou_thres=0.45, device='', classes=None, model_size='default'):
        """
        Faster R-CNN 객체 탐지기 초기화
        
        Args:
            model_path: 모델 가중치 파일 경로 (None이면 기본 가중치 사용)
            conf_thres: 신뢰도 임계값 (0.0 ~ 1.0)
            iou_thres: NMS IoU 임계값 (0.0 ~ 1.0)
            device: 실행 디바이스 ('cpu' 또는 GPU 인덱스)
            classes: 탐지할 클래스 ID 리스트 (None이면 사람만 탐지)
            model_size: 모델 크기 ('default' 또는 'large')
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes if classes is not None else [1]  # 기본값: 사람 클래스만 탐지 (COCO에서 사람은 ID 1)
        
        # 디바이스 설정
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = f'cuda:{device}' if device else 'cuda:0'
        
        # Faster R-CNN 모델 로드
        try:
            logger.info(f"Faster R-CNN 모델 로드 중...")
            
            if model_path and os.path.exists(model_path):
                # 사용자 지정 모델 로드
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"사용자 지정 모델 로드 완료: {model_path}")
            else:
                # 사전 학습된 모델 로드
                self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                logger.info("사전 학습된 Faster R-CNN 모델 로드 완료")
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # 모델 장치 설정
            self.model = self.model.to(self.device)
            
            logger.info(f"Faster R-CNN 모델 로드 완료 (device: {self.device})")
            
        except Exception as e:
            logger.error(f"Faster R-CNN 모델 로드 실패: {e}")
            raise
    
    def detect(self, image):
        """
        이미지에서 객체 탐지 수행
        
        Args:
            image: numpy.ndarray, BGR 이미지 (OpenCV 형식)
            
        Returns:
            detections: 탐지 결과 리스트, 각 항목은 딕셔너리 형태
        """
        # 이미지 변환 (BGR -> RGB, numpy -> tensor)
        image_rgb = np.copy(image)  # Fix negative strides
        image_tensor = F.to_tensor(image_rgb).to(self.device)
        
        # 추론 실행
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # 결과 추출
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # 탐지 결과 필터링 및 변환
        detections = []
        
        for box, score, label in zip(boxes, scores, labels):
            # 신뢰도 임계값 필터링
            if score < self.conf_thres:
                continue
            
            # 클래스 필터링
            if self.classes is not None and label not in self.classes:
                continue
            
            # 박스 좌표 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.astype(float)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(label)
            }
            detections.append(detection)
        
        # NMS (Non-Maximum Suppression) 적용 (필요한 경우)
        if self.iou_thres < 1.0 and len(detections) > 0:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections):
        """간단한 비최대 억제(NMS) 알고리즘 적용"""
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # 면적 계산
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 점수에 따라 내림차순 정렬
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 현재 박스와 나머지 박스 간의 IoU 계산
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # IoU 임계값보다 작은 박스만 남김
            inds = np.where(iou <= self.iou_thres)[0]
            order = order[inds + 1]
        
        # 필터링된 결과 반환
        return [detections[i] for i in keep]