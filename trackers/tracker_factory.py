#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
추적기 팩토리 모듈
지원하는 추적기(DeepSORT 등) 인스턴스를 생성
"""

import os
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def create_tracker(tracker_type, extractor_type='efficient', extractor_size=None, feature_dim=None, 
                  max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30, n_init=3, nn_budget=100, **kwargs):
    """
    추적기 인스턴스 생성
    
    Args:
        tracker_type: 추적기 유형 ('deepsort' 등)
        extractor_type: 특징 추출기 유형 ('simple', 'resnet', 'efficient', 'resnet50', 'efficientnetv2', 'convnext', 'swin')
        extractor_size: 특징 추출기 입력 크기 (None이면 자동 설정)
        feature_dim: 특징 벡터 차원 (None이면 자동 설정)
        max_cosine_distance: 최대 코사인 거리 임계값
        max_iou_distance: 최대 IOU 거리 임계값
        max_age: 소실된 트랙을 유지할 최대 프레임 수
        n_init: 트랙 초기화에 필요한 연속 탐지 수
        nn_budget: 특징 저장소 크기 제한
        **kwargs: 추가 설정
    
    Returns:
        tracker: 추적기 인스턴스
    """
    tracker_type = tracker_type.lower()
    
    if tracker_type == 'deepsort':
        return create_deepsort_tracker(
            extractor_type=extractor_type,
            extractor_size=extractor_size,
            feature_dim=feature_dim,
            max_cosine_distance=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            **kwargs
        )
    
    else:
        raise ValueError(f"지원하지 않는 추적기: {tracker_type}")

def create_deepsort_tracker(extractor_type='efficient', extractor_size=None, feature_dim=None, 
                           max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30, n_init=3, nn_budget=100, **kwargs):
    """
    DeepSORT 추적기 생성
    
    Args:
        extractor_type: 특징 추출기 유형 
        extractor_size: 특징 추출기 입력 크기 (None이면 자동 설정)
        feature_dim: 특징 벡터 차원 (None이면 자동 설정)
        max_cosine_distance: 최대 코사인 거리 임계값
        max_iou_distance: 최대 IOU 거리 임계값
        max_age: 소실된 트랙을 유지할 최대 프레임 수
        n_init: 트랙 초기화에 필요한 연속 탐지 수
        nn_budget: 특징 저장소 크기 제한
        **kwargs: 추가 키워드 인자
        
    Returns:
        생성된 DeepSORT 추적기 인스턴스
    """
    # 특징 추출기 유형에 따른 입력 크기 및 차원 자동 설정
    if extractor_size is None:
        if extractor_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
            extractor_size = 224  # 고급 모델용 큰 입력 크기
        else:
            extractor_size = 128  # 기본 모델용 작은 입력 크기
    
    if feature_dim is None:
        if extractor_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
            feature_dim = 512  # 고급 모델용 큰 특징 차원
        else:
            feature_dim = 128  # 기본 모델용 작은 특징 차원
    
    try:
        from trackers.deep_sort.deep_sort import DeepSORT
        from trackers.deep_sort.core import nn_matching
        from trackers.deep_sort.core.tracker import Tracker as DeepSORTTrackerCore
        
        # 매칭 설정
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # 모델 경로 설정
        if extractor_type == 'resnet50':
            model_path = 'models/resnet50_extractor_pytorch.pth'
        elif extractor_type == 'efficientnetv2':
            model_path = 'models/efficientnetv2_extractor_pytorch.pth'
        elif extractor_type == 'convnext':
            model_path = 'models/convnext_extractor_pytorch.pth'
        elif extractor_type == 'swin':
            model_path = 'models/swin_extractor_pytorch.pth'
        elif extractor_type == 'resnet':
            model_path = 'models/resnet50_extractor_pytorch.pth'
        elif extractor_type == 'efficient':
            model_path = 'models/efficient_extractor_pytorch.pth'
        else:  # simple
            model_path = None
        
        # 현재 디렉토리 기준 모델 경로 보정
        if model_path and not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(script_dir, model_path)
        
        # 특징 추출기 생성
        from trackers.deep_sort.compat_extractor import create_box_encoder
        
        # 입력 크기 설정
        if extractor_type in ['simple', 'resnet', 'efficient']:
            input_shape = (extractor_size, extractor_size // 2, 3)  # 2:1 비율
        else:
            input_shape = (extractor_size, extractor_size, 3)  # 1:1 비율
        
        # 박스 인코더 (특징 추출기) 생성
        encoder = create_box_encoder(
            model_path=model_path, 
            model_type=extractor_type,
            input_shape=input_shape,
            feature_dim=feature_dim
        )
        
        # DeepSORT 트래커 초기화
        tracker = DeepSORT(
            encoder=encoder,
            max_dist=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            **kwargs
        )
        
        logger.info(f"DeepSORT 트래커 생성 완료 (특징 추출기: {extractor_type}, 입력 크기: {extractor_size}, 특징 차원: {feature_dim})")
        logger.info(f"DeepSORT 설정: max_cosine_distance={max_cosine_distance}, max_iou_distance={max_iou_distance}, max_age={max_age}, n_init={n_init}")
        
        return tracker
    
    except ImportError as e:
        logger.error(f"DeepSORT 트래커 생성 실패: {e}")
        sys.exit(1)