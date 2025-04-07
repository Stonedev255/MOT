#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO + DeepSORT 성능 평가 메인 스크립트
다양한 YOLO 모델(YOLOv5, YOLOv7, YOLOv8)과 DeepSORT를 결합하여
MOT16, MOT20 데이터셋에서의 다중 객체 추적 성능을 평가

사용법:
    python main.py --config configs/default.yaml
    python main.py --detector yolov8 --dataset MOT16 --sequence MOT16-14 --save-video
"""

import os
import sys
import time
import argparse
import yaml
import logging
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# 로컬 모듈 임포트
from detectors.detector_factory import create_detector
from trackers.tracker_factory import create_tracker
from utils.metrics import calculate_mot_metrics
from utils.visualization import visualize_tracking_results
from utils.io_utils import load_dataset_sequences, save_results

print("===== 스크립트 실행 시작 =====")
sys.stdout.flush()  # 출력 버퍼 비우기

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='YOLO + DeepSORT 다중 객체 추적 평가')
    
    # 설정 파일
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='설정 파일 경로')
    
    # 탐지기 & 추적기 설정
    parser.add_argument('--detector', type=str, 
                    choices=['yolov5', 'yolov7', 'yolov8', 'faster_rcnn', 'fcos', 'retinanet', 'detr', 'ssd', 'mask_rcnn'], 
                    help='사용할 객체 탐지기 (기본값: 설정 파일에서 로드)')
    parser.add_argument('--tracker', type=str, default='deepsort',
                        help='사용할 객체 추적기 (현재는 deepsort만 지원)')
    
    # 특징 추출기 옵션 추가
    parser.add_argument('--extractor', type=str, default='simple',
                       choices=['simple', 'resnet', 'resnet50', 'efficientnetv2', 'convnext', 'swin'],
                       help='특징 추출기 유형 (simple: 기본 CNN, resnet/resnet50: ResNet50, efficient: EfficientNet, ' +
                           'efficientnetv2: EfficientNetV2, convnext: ConvNeXt, swin: SWIN Transformer)')
    parser.add_argument('--extractor-size', type=int, default=None,
                       help='특징 추출기 입력 크기 (기본값: 모델에 따라 자동 설정)')
    parser.add_argument('--feature-dim', type=int, default=None,
                       help='특징 벡터 차원 (기본값: 모델에 따라 자동 설정)')
    
    # DeepSORT 설정 파라미터 추가
    parser.add_argument('--max-cosine-distance', type=float, default=0.2,
                       help='최대 코사인 거리 임계값 (기본값: 0.2)')
    parser.add_argument('--max-iou-distance', type=float, default=0.7,
                       help='최대 IOU 거리 임계값 (기본값: 0.7)')
    parser.add_argument('--max-age', type=int, default=30,
                       help='트랙 유지 최대 프레임 수 (기본값: 30)')
    parser.add_argument('--n-init', type=int, default=3,
                       help='트랙 초기화 최소 탐지 수 (기본값: 3)')
    parser.add_argument('--nn-budget', type=int, default=100,
                       help='특징 저장소 크기 제한 (기본값: 100)')
    
    # 데이터셋 설정
    parser.add_argument('--dataset', type=str, choices=['MOT16', 'MOT20'],
                        help='평가할 데이터셋 (기본값: 설정 파일에서 로드)')
    parser.add_argument('--sequence', type=str, 
                        help='평가할 시퀀스 (예: MOT16-14, MOT20-08) (기본값: 모든 시퀀스)')
    
    # 출력 설정
    parser.add_argument('--save-video', action='store_true',
                        help='추적 결과 비디오 저장')
    parser.add_argument('--save-txt', action='store_true',
                        help='추적 결과를 텍스트 파일로 저장')
    parser.add_argument('--output-format', type=str, choices=['json', 'csv'], default='json',
                        help='평가 결과 저장 형식 (json 또는 csv)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='결과 저장 디렉토리')
    
    # 탐지 설정
    parser.add_argument('--conf-thres', type=float, 
                        help='객체 탐지 신뢰도 임계값 (기본값: 설정 파일에서 로드)')
    parser.add_argument('--iou-thres', type=float, 
                        help='NMS IoU 임계값 (기본값: 설정 파일에서 로드)')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='',
                        help='사용할 디바이스 (예: cpu, 0, 1, 2, 3) (기본값: 설정 파일에서 로드)')
    
    return parser.parse_args()

def load_config(config_path, args):
    """
    설정 파일 로드 및 명령줄 인자와 병합
    
    Args:
        config_path: 설정 파일 경로
        args: 명령줄 인자
        
    Returns:
        병합된 설정 딕셔너리
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 명령줄 인자로 제공된 값으로 설정 덮어쓰기
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            key_name = key
            # 명령행에서 - 대신 _를 사용하는 인자 처리 (e.g., save-video -> save_video)
            if key == 'save_video':
                key_name = 'save_video'
            elif key == 'save_txt':
                key_name = 'save_txt'
            elif key == 'conf_thres':
                key_name = 'conf_thres'
            elif key == 'iou_thres':
                key_name = 'iou_thres'
            elif key == 'output_format':
                key_name = 'output_format'
            elif key == 'output_dir':
                key_name = 'output_dir'
            elif key == 'extractor':
                key_name = 'extractor'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'extractor_size':
                key_name = 'extractor_size'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'feature_dim':
                key_name = 'feature_dim'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'max_cosine_distance':
                key_name = 'max_cosine_distance'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'max_iou_distance':
                key_name = 'max_iou_distance'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'max_age':
                key_name = 'max_age'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'n_init':
                key_name = 'n_init'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'nn_budget':
                key_name = 'nn_budget'
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            
            if key_name in config:
                config[key_name] = value
                print(f"설정 업데이트: {key_name} -> {value}")
            elif key == 'sequence':
                # sequence 인자 명시적 처리 - 문자열로 설정
                config['sequence'] = value
                print(f"시퀀스 설정: {value}")
    
    return config

# setup_environment 함수를 수정하여 결과를 모델별로 구분하여 저장합니다.
def setup_environment(config):
    """
    필요한 환경 설정 및 디렉토리 생성
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        업데이트된 설정 딕셔너리
    """
    # 결과 디렉토리 생성
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 디텍터 및 데이터셋별 하위 디렉토리 생성
    dataset_dir = output_dir / config['dataset'].lower()
    detector_dir = dataset_dir / config['detector']
    
    # 특징 추출기 유형별 디렉토리 추가
    extractor_type = config.get('extractor', 'efficient')
    extractor_dir = detector_dir / extractor_type
    
    for subdir in ['metrics', 'visualizations', 'tracks']:
        (extractor_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 업데이트된 경로 설정 저장
    config['paths'] = {
        'output_dir': str(output_dir),
        'dataset_dir': str(dataset_dir),
        'detector_dir': str(detector_dir),
        'extractor_dir': str(extractor_dir),
        'metrics_dir': str(extractor_dir / 'metrics'),
        'visualizations_dir': str(extractor_dir / 'visualizations'),
        'tracks_dir': str(extractor_dir / 'tracks')
    }
    
    return config

def process_sequence(sequence, detector, tracker, config):
    """
    단일 시퀀스에 대한 객체 탐지 및 추적 수행
    
    Args:
        sequence: 시퀀스 정보 딕셔너리 (경로, 이미지 파일 목록 등)
        detector: 객체 탐지기 인스턴스
        tracker: 객체 추적기 인스턴스
        config: 설정 딕셔너리
        
    Returns:
        추적 결과 및 성능 지표
    """
    logger.info(f"Processing sequence: {sequence['name']}")
    
    # 시퀀스 정보 출력
    logger.info(f"  - Frames: {len(sequence['images'])}")
    logger.info(f"  - Image path: {sequence['image_path']}")
    
    # 비디오 작성기 설정 (결과 시각화용)
    if config['save_video']:
        video_path = os.path.join(config['paths']['visualizations_dir'], f"{sequence['name']}.mp4")
        
        # 첫 이미지를 읽어서 해상도 확인
        first_img = cv2.imread(sequence['images'][0])
        height, width = first_img.shape[:2]
        
        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,  # FPS
            (width, height)
        )
    else:
        video_writer = None
    
    # 추적 결과 및 시간 측정을 위한 변수
    tracking_results = []
    frame_times = []
    
    # 시퀀스의 각 프레임 처리
    for frame_idx, img_path in enumerate(tqdm(sequence['images'], desc=f"Tracking {sequence['name']}")):
        # 이미지 로드
        frame = cv2.imread(img_path)
        
        # 시간 측정 시작
        start_time = time.time()
        
        # 객체 탐지
        detections = detector.detect(frame)
        
        # 객체 추적
        tracks = tracker.update(detections, frame)
        
        # 시간 측정 종료
        end_time = time.time()
        frame_times.append(end_time - start_time)
        
        # 추적 결과 저장
        for track in tracks:
            # 포맷: [frame_idx, track_id, x1, y1, x2, y2, confidence]
            result = [
                frame_idx + 1,  # MOT Challenge 형식은 1부터 시작
                track['track_id'],
                *track['bbox'],  # x1, y1, x2, y2
                track['confidence']
            ]
            tracking_results.append(result)
        
        # 결과 시각화
        if video_writer is not None:
            vis_frame = visualize_tracking_results(frame.copy(), tracks, frame_idx)
            video_writer.write(vis_frame)
    
    # 비디오 작성기 종료
    if video_writer is not None:
        video_writer.release()
    
    # FPS 계산
    fps = 1.0 / np.mean(frame_times) if frame_times else 0
    logger.info(f"  - Average FPS: {fps:.2f}")
    
    # 추적 결과를 MOT Challenge 형식의 텍스트 파일로 저장
    if config['save_txt']:
        result_path = os.path.join(config['paths']['tracks_dir'], f"{sequence['name']}.txt")
        
        with open(result_path, 'w') as f:
            for result in tracking_results:
                # MOT Challenge 형식: frame, id, x, y, w, h, confidence, -1, -1, -1
                frame_id, track_id, x1, y1, x2, y2, confidence = result
                w, h = x2 - x1, y2 - y1
                f.write(f"{int(frame_id)},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{confidence:.6f},-1,-1,-1\n")
    
    # 평가 지표 계산 (GT가 있는 경우)
    metrics = {}
    if sequence['gt_path'] and os.path.exists(sequence['gt_path']):
        try:
            # 메트릭 계산
            metrics = calculate_mot_metrics(tracking_results, sequence['gt_path'], sequence['name'])
            metrics['fps'] = fps

            logger.info("\n===== 성능 평가 결과 =====")
            logger.info(f"MOTA: {metrics['mota']:.2f}%")
            logger.info(f"MOTP: {metrics['motp']:.2f}%")
            logger.info(f"IDF1: {metrics['idf1']:.2f}%")  # 추가
            logger.info(f"Precision: {metrics['precision']:.2f}%")  # 추가
            logger.info(f"Recall: {metrics['recall']:.2f}%")  # 추가
            logger.info(f"ID Switches: {metrics['num_switches']}")
            logger.info(f"Fragmentations: {metrics['num_fragmentations']}")  # 추가
            logger.info(f"FP (오검지): {metrics['fp']}")  # 추가
            logger.info(f"FN (미검지): {metrics['fn']}")  # 추가
            logger.info(f"MT: {metrics['mostly_tracked']}")
            logger.info(f"ML: {metrics['mostly_lost']}")
            logger.info(f"FPS: {fps:.2f}")
            logger.info("==========================\n")
        except Exception as e:
            logger.error(f"지표 계산 중 오류 발생: {e}")
            metrics = {
                'mota': 0.0,
                'motp': 0.0,
                'num_switches': 0,
                'mostly_tracked': 0,
                'mostly_lost': 0,
                'num_fragmentations': 0,  # 수정: fragmenttations → num_fragmentations
                'idf1': 0.0,              # 추가
                'precision': 0.0,         # 추가
                'recall': 0.0,            # 추가
                'fp': 0,                  # 추가
                'fn': 0,                  # 추가
                'fps': fps
            }
    else:
        logger.warning(f"GT 데이터를 찾을 수 없음: {sequence['gt_path']}")
        metrics = {
            'mota': 0.0,
            'motp': 0.0,
            'num_switches': 0,
            'mostly_tracked': 0,
            'mostly_lost': 0,
            'num_fragmentations': 0,     # 추가
            'idf1': 0.0,                 # 추가
            'precision': 0.0,            # 추가
            'recall': 0.0,               # 추가
            'fp': 0,                     # 추가
            'fn': 0,                     # 추가
            'fps': fps
        }

    return tracking_results, metrics

def main():
    """메인 실행 함수"""
    # 인자 파싱
    args = parse_args()
    
    # 설정 로드
    config = load_config(args.config, args)
    
    # 환경 설정
    config = setup_environment(config)
    
    logger.info("===== YOLO + DeepSORT 성능 평가 시작 =====")
    logger.info(f"Detector: {config['detector']}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Sequence: {config.get('sequence', 'All')}")
    logger.info(f"Extractor: {config.get('extractor', 'efficient')}")
    logger.info(f"결과 저장 경로: {config['paths']['extractor_dir']}")
    
    # 객체 탐지기 생성
    detector = create_detector(
        config['detector'],
        conf_thres=config['conf_thres'],
        iou_thres=config['iou_thres'],
        device=config['device']
    )
    
    # 객체 추적기 생성 (특징 추출기 유형 및 DeepSORT 파라미터 전달)
    tracker_config = {
        'extractor_type': config.get('extractor', 'efficient'),
        'extractor_size': config.get('extractor_size'),
        'feature_dim': config.get('feature_dim'),
        'max_cosine_distance': config.get('max_cosine_distance', 0.2),
        'max_iou_distance': config.get('max_iou_distance', 0.7),
        'max_age': config.get('max_age', 30),
        'n_init': config.get('n_init', 3),
        'nn_budget': config.get('nn_budget', 100)
    }
    
    tracker = create_tracker(
        config['tracker'],
        **tracker_config
    )
    
    # 데이터셋 시퀀스 로드
    sequences = load_dataset_sequences(
        config['dataset'], 
        config['data_dir'],
        sequence_name=config.get('sequence')
    )
    
    logger.info(f"총 {len(sequences)}개 시퀀스 처리 예정")
    
    # 각 시퀀스 처리
    all_results = {}
    for sequence in sequences:
        tracking_results, metrics = process_sequence(sequence, detector, tracker, config)
        
        # 결과 저장
        sequence_name = sequence['name']
        # 결과 저장 부분
        result_data = {
            'detector': config['detector'],
            'dataset': config['dataset'],
            'sequence': sequence_name,
            'MOTA': metrics['mota'],
            'MOTP': metrics['motp'],
            'IDF1': metrics['idf1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'MT': metrics['mostly_tracked'],
            'ML': metrics['mostly_lost'],
            'ID Switches': metrics['num_switches'],
            'Fragmentations': metrics['num_fragmentations'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'FPS': metrics['fps']
        }
        all_results[sequence_name] = result_data
        
        # 개별 시퀀스 결과 저장
        save_results(
            result_data,
            os.path.join(config['paths']['metrics_dir'], f"{sequence_name}.{config['output_format']}"),
            format=config['output_format']
        )
    
    # 전체 결과 요약 저장
    summary_path = os.path.join(config['paths']['metrics_dir'], f"summary.{config['output_format']}")
    save_results(all_results, summary_path, format=config['output_format'])
    
    logger.info(f"모든 결과가 {config['paths']['metrics_dir']}에 저장되었습니다.")
    logger.info(f"특징 추출기: {config.get('extractor', 'efficient')}")  # 이 줄 추가
    logger.info("===== YOLO + DeepSORT 성능 평가 완료 =====")

if __name__ == "__main__":
    main()