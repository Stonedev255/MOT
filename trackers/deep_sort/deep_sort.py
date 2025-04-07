#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSORT 추적기 클래스
객체 탐지 결과를 DeepSORT 알고리즘으로 추적
"""

import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DeepSORT:
    """DeepSORT 추적기 클래스"""
    
    def __init__(self, encoder, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        """
        DeepSORT 추적기 초기화
        
        Args:
            encoder: 특징 추출 함수
            max_dist: 최대 코사인 거리 (임계값)
            min_confidence: 최소 탐지 신뢰도
            nms_max_overlap: NMS 최대 중첩 임계값
            max_iou_distance: 최대 IoU 거리 (임계값)
            max_age: 트랙이 유지되는 최대 프레임 수
            n_init: 트랙 초기화에 필요한 연속 탐지 수
            nn_budget: 특징 저장소 크기 제한
            use_cuda: CUDA 사용 여부
        """
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.encoder = encoder  # 특징 추출기
        
        # DeepSORT 환경 설정
        self._setup_deepsort()
        
        # DeepSORT 트래커 초기화
        from .core import nn_matching
        from .core.tracker import Tracker
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
    
    def _setup_deepsort(self):
        """DeepSORT 환경 설정"""
        # 현재 프로젝트에서 DeepSORT core 모듈 사용
        # 외부 저장소 클론이 필요 없음
        
        # 모듈 구조 확인
        core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core')
        
        # core 디렉토리가 존재하는지 확인
        if not os.path.exists(core_dir):
            logger.error(f"DeepSORT core 디렉토리가 존재하지 않습니다: {core_dir}")
            raise FileNotFoundError(f"DeepSORT core directory not found: {core_dir}")
        
        logger.info(f"DeepSORT core 모듈 로드 완료: {core_dir}")
    
    def _get_features(self, bbox_tlwh, ori_img):
        """
        주어진 객체 영역의 특징 추출
        
        Args:
            bbox_tlwh: numpy.ndarray, [N, 4], (top, left, width, height) 형식
            ori_img: numpy.ndarray, 원본 이미지
            
        Returns:
            features: numpy.ndarray, [N, 128/512], 추출된 특징 벡터
        """
        features = self.encoder(ori_img, bbox_tlwh)
        return features
    
    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        (top, left, width, height) -> (x1, y1, x2, y2) 변환
        
        Args:
            bbox_tlwh: numpy.ndarray, [N, 4], (top, left, width, height) 형식
            
        Returns:
            bbox_xyxy: numpy.ndarray, [N, 4], (x1, y1, x2, y2) 형식
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = max(int(x+w), 0)
        y2 = max(int(y+h), 0)
        return x1, y1, x2, y2
    
    def _xyxy_to_tlwh(self, bbox_xyxy):
        """
        (x1, y1, x2, y2) -> (top, left, width, height) 변환
        
        Args:
            bbox_xyxy: numpy.ndarray, [N, 4], (x1, y1, x2, y2) 형식
            
        Returns:
            bbox_tlwh: numpy.ndarray, [N, 4], (top, left, width, height) 형식
        """
        x1, y1, x2, y2 = bbox_xyxy
        t = y1
        l = x1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h
    
    def update(self, detections, image):
        """
        객체 탐지 결과를 기반으로 추적 수행
        
        Args:
            detections: 객체 탐지 결과 리스트, 각 항목은 {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int} 형식
            image: numpy.ndarray, BGR 이미지 (OpenCV 형식)
            
        Returns:
            tracks: 추적 결과 리스트, 각 항목은 {'track_id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float} 형식
        """
        if not detections:
            # 탐지 결과가 없으면 빈 리스트 반환
            self.tracker.predict()
            return []
        
        # 탐지 결과 필터링 및 변환
        bbox_xyxy = []
        confidences = []
        for detection in detections:
            if detection['confidence'] < self.min_confidence:
                continue
            
            bbox_xyxy.append(detection['bbox'])
            confidences.append(detection['confidence'])
        
        if not bbox_xyxy:
            # 필터링 후 남은 탐지 결과가 없으면 빈 리스트 반환
            self.tracker.predict()
            return []
        
        # numpy 배열로 변환
        bbox_xyxy = np.array(bbox_xyxy)
        confidences = np.array(confidences)
        
        # (x1, y1, x2, y2) -> (top, left, width, height) 변환
        bbox_tlwh = []
        for x1, y1, x2, y2 in bbox_xyxy:
            t = y1
            l = x1
            w = int(x2 - x1)
            h = int(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            bbox_tlwh.append([t, l, w, h])
        
        if not bbox_tlwh:
            # 유효한 바운딩 박스가 없으면 빈 리스트 반환
            self.tracker.predict()
            return []
        
        # 특징 추출
        features = self._get_features(np.array(bbox_tlwh), image)
        
        # DeepSORT 탐지 객체 생성
        from .core.detection import Detection
        detections_ds = []
        for i, (bbox, conf, feature) in enumerate(zip(bbox_tlwh, confidences, features)):
            detections_ds.append(Detection(bbox, conf, feature))
        
        # 추적 수행
        self.tracker.predict()
        self.tracker.update(detections_ds)
        
        # 추적 결과 변환 (모든 추적 객체 포함)
        tracks = []
        for track in self.tracker.tracks:
            # 모든 트랙 반환(확정 여부와 상관없이)
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            
            box = track.to_tlwh()  # (top, left, width, height)
            x1, y1 = box[1], box[0]
            x2, y2 = box[1] + box[2], box[0] + box[3]
            track_id = track.track_id
            
            tracks.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': 1.0,
                'confirmed': track.is_confirmed()  # 트랙 확정 상태 추가
            })
        
        return tracks