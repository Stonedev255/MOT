#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8 객체 탐지기 구현
ultralytics 패키지를 사용하여 YOLOv8 모델로 객체 탐지 수행
"""

import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLOv8Detector:
    """YOLOv8 객체 탐지기 클래스"""
    
    def __init__(self, model_path=None, conf_thres=0.5, iou_thres=0.45, device='', classes=None):
        """
        YOLOv8 객체 탐지기 초기화
        
        Args:
            model_path: 모델 가중치 파일 경로 (None이면 기본 가중치 사용)
            conf_thres: 신뢰도 임계값 (0.0 ~ 1.0)
            iou_thres: NMS IoU 임계값 (0.0 ~ 1.0)
            device: 실행 디바이스 ('cpu' 또는 GPU 인덱스)
            classes: 탐지할 클래스 ID 리스트 (None이면 사람만 탐지)
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes if classes is not None else [0]  # 기본값: 사람 클래스만 탐지
        
        # 디바이스 설정
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = f'cuda:{device}' if device else 'cuda:0'
        
        # YOLOv8 모델 로드
        try:
            from ultralytics import YOLO
            
            if not model_path:
                model_path = 'yolov8x.pt'  # 기본 가중치
                
            logger.info(f"YOLOv8 모델 로드 중: {model_path}")
            self.model = YOLO(model_path)
            
            # 디바이스 설정
            self.model.to(self.device)
            
            logger.info(f"YOLOv8 모델 로드 완료 (device: {self.device})")
        
        except ImportError:
            logger.error("ultralytics 패키지를 설치해야 합니다. 설치 중...")
            os.system("pip install ultralytics")
            from ultralytics import YOLO
            
            if not model_path:
                model_path = 'yolov8x.pt'  # 기본 가중치
                
            logger.info(f"YOLOv8 모델 로드 중: {model_path}")
            self.model = YOLO(model_path)
            
            # 디바이스 설정
            self.model.to(self.device)
            
            logger.info(f"YOLOv8 모델 로드 완료 (device: {self.device})")
    
    def detect(self, image):
        """
        이미지에서 객체 탐지 수행
        
        Args:
            image: numpy.ndarray, BGR 이미지 (OpenCV 형식)
            
        Returns:
            detections: 탐지 결과 리스트, 각 항목은 [x1, y1, x2, y2, confidence, class_id] 형식
        """
        # 추론 실행
        results = self.model(
            image, 
            conf=self.conf_thres,
            iou=self.iou_thres,
            classes=self.classes,
            verbose=False
        )
        
        # 탐지 결과 변환 및 반환
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls = int(box.cls.item())
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': cls
                }
                detections.append(detection)
        
        return detections