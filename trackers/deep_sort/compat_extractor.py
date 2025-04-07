#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch 기반 DeepSORT 특징 추출기
"""

import os
import numpy as np
import cv2
import torch
import logging
from torchvision import transforms

logger = logging.getLogger(__name__)

class PyTorchFeatureExtractor:
    """PyTorch 기반 특징 추출기"""
    
    def __init__(self, model_path=None, model_type=None, input_shape=(128, 64, 3), feature_dim=128):
        """
        특징 추출기 초기화
        
        Args:
            model_path: 모델 파일 경로 (없으면 기본 모델 생성)
            model_type: 모델 유형 (resnet, efficient, swin, convnext, efficientnetv2)
            input_shape: 입력 이미지 크기 (높이, 너비, 채널)
            feature_dim: 특징 차원
        """
        # 모델 유형 자동 감지
        if model_type is None and model_path is not None:
            if 'resnet' in model_path:
                model_type = 'resnet'
            elif 'efficient' in model_path and 'v2' in model_path:
                model_type = 'efficientnetv2'
            elif 'efficient' in model_path:
                model_type = 'efficient'
            elif 'swin' in model_path:
                model_type = 'swin'
            elif 'convnext' in model_path:
                model_type = 'convnext'
        
        self.model_type = model_type if model_type else 'simple'
        
        # 고급 모델에 대한 입력 크기 자동 조정
        if self.model_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
            self.input_shape = (224, 224, 3)
        else:
            self.input_shape = input_shape
        
        # 특징 차원 설정
        self.feature_dim = feature_dim
        
        # 상대 경로 사용
        if model_path is None:
            # 현재 파일 위치 기준으로 모델 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
            self.model_path = os.path.join(root_dir, 'models', f'{self.model_type}_extractor_pytorch.pth')
        else:
            self.model_path = model_path
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
        
        # 이미지 전처리를 위한 변환 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """모델 로드 또는 생성"""
        try:
            # 파일 확장자 확인
            _, ext = os.path.splitext(self.model_path)
                
            # .pth 파일 (상태 사전)인 경우
            if ext == '.pth':
                logger.info(f"PyTorch 모델 상태 사전 로드 중: {self.model_path}")

                # 모델 파일이 존재하는지 확인
                if os.path.exists(self.model_path):
                    # 모델 유형에 따라 적절한 모델 생성
                    if self.model_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
                        self.model = self._create_advanced_model()
                    else:
                        self.model = self._create_simple_model()
                    
                    # 상태 사전 로드
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    logger.info(f"상태 사전 로드 완료: {self.model_path}")
                else:
                    # 파일이 없는 경우 모델 유형에 따라 자동 생성
                    logger.warning(f"상태 사전 파일이 존재하지 않음: {self.model_path}")
                    logger.info(f"{self.model_type} 모델 자동 생성 시작...")
                    
                    if self.model_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
                        self.model = self._create_advanced_model()
                    else:
                        self.model = self._create_simple_model()
                    
                    # 모델 저장
                    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.model_path)
                    logger.info(f"{self.model_type} 모델 생성 및 저장 완료: {self.model_path}")
                
            # .pt 파일 (전체 모델)인 경우
            elif ext == '.pt':
                # 스크립트 모델인지 확인
                if 'script' in self.model_path:
                    logger.info(f"TorchScript 모델 로드 중: {self.model_path}")
                    self.model = torch.jit.load(self.model_path, map_location=self.device)
                else:
                    logger.info(f"PyTorch 모델 로드 중: {self.model_path}")
                    self.model = torch.load(self.model_path, map_location=self.device)
                logger.info(f"모델 로드 완료: {self.model_path}")
                
            # 파일이 존재하지 않는 경우
            else:
                logger.info(f"모델 파일이 없습니다. {self.model_type} 특징 추출 모델을 생성합니다...")
                
                if self.model_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
                    self.model = self._create_advanced_model()
                else:
                    self.model = self._create_simple_model()
                
                # 모델 저장
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path.replace('.pt', '.pth'))
                logger.info(f"{self.model_type} 모델 생성 및 저장 완료: {self.model_path}")
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # 모델을 장치로 이동
            self.model = self.model.to(self.device)
            
            # 더미 입력으로 모델 테스트
            logger.info("모델 테스트 중...")
            dummy_input = torch.zeros((1, 3, self.input_shape[0], self.input_shape[1]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
                logger.info(f"테스트 출력 크기: {output.shape}")
            
            logger.info(f"PyTorch 모델 로드 및 테스트 완료: {self.model_type}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def _create_advanced_model(self):
        """고급 특징 추출 모델 생성 (SWIN, ConvNeXt, EfficientNetV2)"""
        try:
            import torchvision.models as models
            import torch.nn as nn
            
            if self.model_type == 'swin':
                logger.info("SWIN Transformer 모델 생성 중...")
                # SWIN Transformer
                base_model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
                num_features = base_model.head.in_features
                base_model.head = nn.Identity()
                
                # 특징 추출 헤드
                head = nn.Sequential(
                    nn.BatchNorm1d(num_features),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, self.feature_dim)
                )
                
                model = nn.Sequential(base_model, head)
                
            elif self.model_type == 'convnext':
                logger.info("ConvNeXt 모델 생성 중...")
                # ConvNeXt
                base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                
                # 분류기 완전히 수정 - Flatten 추가 및 레이어 재구성
                base_model.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                num_features = 768  # ConvNeXt-Tiny의 특징 채널 수
                
                # 특징 추출 헤드
                head = nn.Sequential(
                    nn.BatchNorm1d(num_features),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, self.feature_dim)
                )
                
                model = nn.Sequential(base_model, head)
                
            elif self.model_type == 'efficientnetv2':
                logger.info("EfficientNetV2 모델 생성 중...")
                # EfficientNetV2
                base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
                num_features = base_model.classifier[1].in_features
                base_model.classifier = nn.Identity()
                
                # 특징 추출 헤드
                head = nn.Sequential(
                    nn.BatchNorm1d(num_features),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, self.feature_dim)
                )
                
                model = nn.Sequential(base_model, head)
                
            elif self.model_type == 'resnet50':
                logger.info("ResNet50 모델 생성 중...")
                # ResNet50
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                num_features = base_model.fc.in_features
                base_model.fc = nn.Identity()
                
                # 특징 추출 헤드
                head = nn.Sequential(
                    nn.BatchNorm1d(num_features),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, self.feature_dim)
                )
                
                model = nn.Sequential(base_model, head)
                
            else:
                logger.warning(f"지원되지 않는 고급 모델 유형: {self.model_type}")
                model = self._create_simple_model()
                
            return model
            
        except Exception as e:
            logger.error(f"고급 모델 생성 실패: {e}")
            logger.info("기본 CNN 모델을 대신 사용합니다.")
            return self._create_simple_model()
            
    def _create_simple_model(self):
        """간단한 CNN 특징 추출 모델 생성"""
        import torch.nn as nn
        
        logger.info("기본 CNN 모델 생성 중...")
        
        # 입력 크기에 따라 최종 특징 크기 계산
        h, w = self.input_shape[:2]
        
        # 3번의 풀링 후 최종 특징 크기 계산
        final_h, final_w = h // 8, w // 8
        final_features = 128 * final_h * final_w
        
        model = nn.Sequential(
            # 입력: 3xHxW
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 32x(H/2)x(W/2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64x(H/4)x(W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 128x(H/8)x(W/8)
            nn.Flatten(),
            # 128*(H/8)*(W/8)
            nn.Linear(final_features, self.feature_dim)
        )
        
        return model
    
    def __call__(self, image, boxes):
        """
        이미지에서 주어진 박스 영역의 특징 추출
        
        Args:
            image: numpy.ndarray, BGR 이미지 (OpenCV 형식)
            boxes: numpy.ndarray, [N, 4], (top, left, width, height) 형식
            
        Returns:
            features: numpy.ndarray, [N, feature_dim], 추출된 특징 벡터
        """
        if len(boxes) == 0:
            return np.array([])
        
        image_patches = []
        
        for box in boxes:
            # 박스 좌표 추출
            top, left, width, height = box
            
            # 이미지 크롭
            try:
                crop = image[int(top):int(top+height), int(left):int(left+width)]
                
                if crop.size == 0:
                    # 유효하지 않은 크롭 처리
                    logger.warning(f"빈 크롭 발생: {box}")
                    crop = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
            except Exception as e:
                logger.warning(f"이미지 크롭 실패: {e}, 박스: {box}")
                crop = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
            
            # RGB로 변환
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # 크기 조정
            crop = cv2.resize(crop, (self.input_shape[1], self.input_shape[0]))
            
            # PyTorch 텐서로 변환 및 정규화
            tensor = self.transform(crop).unsqueeze(0)
            image_patches.append(tensor)
        
        # 배치로 구성
        if len(image_patches) > 0:
            image_batch = torch.cat(image_patches).to(self.device)
            
            # 특징 추출 (평가 모드에서)
            with torch.no_grad():
                features = self.model(image_batch)
            
            # GPU 메모리에서 CPU로 이동
            features = features.cpu().numpy()
            
            # 정규화
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            return features
        else:
            return np.array([])

def create_box_encoder(model_path=None, batch_size=32, extractor_fn=None, model_type=None, input_shape=None, feature_dim=None):
    """
    특징 추출기 생성 함수 (DeepSORT와 호환)
    
    Args:
        model_path: 모델 파일 경로 (None이면 기본 경로 사용)
        batch_size: 배치 크기 (사용하지 않음)
        extractor_fn: 외부에서 제공된 특징 추출 함수 (있으면 사용)
        model_type: 모델 유형 (resnet, efficient, swin, convnext, efficientnetv2)
        input_shape: 입력 이미지 크기 (높이, 너비, 채널)
        feature_dim: 특징 벡터 차원
        
    Returns:
        encoder: 특징 추출 함수
    """
    # 외부에서 제공된 추출기 함수가 있으면 사용
    if extractor_fn is not None:
        logger.info("외부 제공 특징 추출기 사용")
        return extractor_fn
    
    # 모델 유형에 따라 입력 크기 및 특징 차원 자동 설정
    if model_type in ['swin', 'convnext', 'efficientnetv2', 'resnet50']:
        if input_shape is None:
            input_shape = (224, 224, 3)
        if feature_dim is None:
            feature_dim = 512
    else:
        if input_shape is None:
            input_shape = (128, 64, 3)
        if feature_dim is None:
            feature_dim = 128
    
    # 모델 경로 확인
    if model_path is None:
        # 현재 파일 위치 기준으로 모델 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # 모델 유형에 따른 기본 경로 설정
        if model_type is None:
            model_type = 'efficient'  # 기본값
        
        model_path = os.path.join(root_dir, 'models', f'{model_type}_extractor_pytorch.pth')
    
    # 모델 파일 확장자 확인
    _, ext = os.path.splitext(model_path)
    
    # 모델이 TorchScript인 경우
    if ext == '.pt' and 'script' in model_path:
        try:
            # PyTorchScriptFeatureExtractor 클래스 정의가 필요함
            class PyTorchScriptFeatureExtractor:
                def __init__(self, model_path):
                    self.model = torch.jit.load(model_path)
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.to(self.device)
                    
                def __call__(self, image, boxes):
                    # 구현이 필요함
                    pass
            
            extractor = PyTorchScriptFeatureExtractor(model_path)
        except NameError:
            logger.warning("PyTorchScriptFeatureExtractor 클래스가 정의되지 않았습니다. PyTorchFeatureExtractor를 사용합니다.")
            extractor = PyTorchFeatureExtractor(model_path, model_type, input_shape, feature_dim)
    # 일반 PyTorch 모델인 경우
    else:
        extractor = PyTorchFeatureExtractor(model_path, model_type, input_shape, feature_dim)
    
    return extractor