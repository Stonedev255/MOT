#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
객체 탐지기 팩토리 모듈
지원하는 객체 탐지기(YOLOv5, YOLOv7, YOLOv8, FCOS, Faster R-CNN, RetinaNet, DETR, SSD, Mask R-CNN) 인스턴스를 생성
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_detector(detector_type, **kwargs):
    """
    객체 탐지기 인스턴스 생성
    
    Args:
        detector_type: 탐지기 유형 ('yolov5', 'yolov7', 'yolov8', 'fcos', 'faster_rcnn', 'retinanet', 'detr', 'ssd', 'mask_rcnn')
        **kwargs: 추가 설정 (conf_thres, iou_thres, device 등)
    
    Returns:
        detector: 객체 탐지기 인스턴스
    """
    detector_type = detector_type.lower()
    
    try:
        import ultralytics
        
        if detector_type == 'yolov5':
            from .yolov5_detector import YOLOv5Detector
            return YOLOv5Detector(**kwargs)
        elif detector_type == 'yolov7':
            from .yolov7_detector import YOLOv7Detector
            return YOLOv7Detector(**kwargs)
        elif detector_type == 'yolov8':
            from .yolov8_detector import YOLOv8Detector
            return YOLOv8Detector(**kwargs)
        elif detector_type == 'faster_rcnn':
            from .faster_rcnn_detector import FasterRCNNDetector
            return FasterRCNNDetector(**kwargs)
        elif detector_type == 'fcos':
            from .fcos_detector import FCOSDetector
            return FCOSDetector(**kwargs)
        elif detector_type == 'retinanet':
            from .retinanet_detector import RetinaNetDetector
            return RetinaNetDetector(**kwargs)
        elif detector_type == 'detr':
            from .detr_detector import DETRDetector
            return DETRDetector(**kwargs)
        elif detector_type == 'ssd':
            from .ssd_detector import SSDDetector
            return SSDDetector(**kwargs)
        elif detector_type == 'mask_rcnn':
            from .mask_rcnn_detector import MaskRCNNDetector
            return MaskRCNNDetector(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 객체 탐지기: {detector_type}")
    
    except ImportError:
        if detector_type in ['yolov5', 'yolov7', 'yolov8']:
            os.system("pip install ultralytics")
        if detector_type in ['faster_rcnn', 'fcos', 'retinanet', 'detr', 'ssd', 'mask_rcnn']:
            os.system("pip install torch torchvision")
        
        if detector_type == 'yolov5':
            from .yolov5_detector import YOLOv5Detector
            return YOLOv5Detector(**kwargs)
        elif detector_type == 'yolov7':
            from .yolov7_detector import YOLOv7Detector
            return YOLOv7Detector(**kwargs)
        elif detector_type == 'yolov8':
            from .yolov8_detector import YOLOv8Detector
            return YOLOv8Detector(**kwargs)
        elif detector_type == 'faster_rcnn':
            from .faster_rcnn_detector import FasterRCNNDetector
            return FasterRCNNDetector(**kwargs)
        elif detector_type == 'fcos':
            from .fcos_detector import FCOSDetector
            return FCOSDetector(**kwargs)
        elif detector_type == 'retinanet':
            from .retinanet_detector import RetinaNetDetector
            return RetinaNetDetector(**kwargs)
        elif detector_type == 'detr':
            from .detr_detector import DETRDetector
            return DETRDetector(**kwargs)
        elif detector_type == 'ssd':
            from .ssd_detector import SSDDetector
            return SSDDetector(**kwargs)
        elif detector_type == 'mask_rcnn':
            from .mask_rcnn_detector import MaskRCNNDetector
            return MaskRCNNDetector(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 객체 탐지기: {detector_type}")
    
    except Exception as e:
        logger.error(f"탐지기 생성 중 오류 발생: {e}")
        raise