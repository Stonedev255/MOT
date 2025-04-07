#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
입출력 유틸리티 함수 모듈
데이터셋 로드 및 결과 저장 등의 기능 제공
"""

import os
import sys
import json
import csv
import logging
import glob
from pathlib import Path

logger = logging.getLogger(__name__)

def load_dataset_sequences(dataset_name, data_dir, sequence_name=None):
    """
    데이터셋 시퀀스 정보 로드
    
    Args:
        dataset_name: 데이터셋 이름 ('MOT16' 또는 'MOT20')
        data_dir: 데이터 디렉토리 경로
        sequence_name: 특정 시퀀스 이름 (None이면 모든 시퀀스)
        
    Returns:
        sequences: 시퀀스 정보 리스트
    """
    dataset_name = dataset_name.upper()
    dataset_path = os.path.join(data_dir, dataset_name, 'train')
    
    # 데이터셋 경로가 존재하는지 확인
    if not os.path.exists(dataset_path):
        logger.error(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # 시퀀스 목록 가져오기
    if sequence_name:
        logger.info(f"지정된 시퀀스: {sequence_name}")
        sequence_paths = [os.path.join(dataset_path, sequence_name)]
        if not os.path.exists(sequence_paths[0]):
            logger.error(f"시퀀스 경로가 존재하지 않습니다: {sequence_paths[0]}")
            raise FileNotFoundError(f"Sequence path not found: {sequence_paths[0]}")
    else:
        logger.info("시퀀스 미지정: 모든 시퀀스 로드")
        sequence_paths = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))]
        sequence_paths.sort()
    
    sequences = []
    for seq_path in sequence_paths:
        seq_name = os.path.basename(seq_path)
        img_path = os.path.join(seq_path, 'img1')
        det_path = os.path.join(seq_path, 'det', 'det.txt')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        
        # 이미지 디렉토리 확인
        if not os.path.exists(img_path):
            logger.warning(f"이미지 디렉토리가 존재하지 않습니다: {img_path}")
            continue
        
        # 이미지 파일 목록 가져오기
        img_files = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
        if not img_files:
            logger.warning(f"이미지 파일이 없습니다: {img_path}")
            continue
        
        # 시퀀스 정보 저장
        sequence = {
            'name': seq_name,
            'path': seq_path,
            'image_path': img_path,
            'det_path': det_path if os.path.exists(det_path) else None,
            'gt_path': gt_path if os.path.exists(gt_path) else None,
            'images': img_files
        }
        sequences.append(sequence)
        
        logger.info(f"시퀀스 로드: {seq_name} ({len(img_files)} frames)")
    
    return sequences

def save_results(results, output_path, format='json'):
    """
    평가 결과 저장
    
    Args:
        results: 저장할 결과 데이터 (딕셔너리 또는 리스트)
        output_path: 출력 파일 경로
        format: 저장 형식 ('json' 또는 'csv')
    """
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.lower() == 'json':
        # JSON 형식으로 저장
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    elif format.lower() == 'csv':
        # CSV 형식으로 저장
        if isinstance(results, dict):
            # 딕셔너리를 리스트로 변환 (각 시퀀스별 결과)
            if all(isinstance(v, dict) for v in results.values()):
                # 시퀀스별 결과인 경우
                data = [{'sequence': k, **v} for k, v in results.items()]
            else:
                # 단일 결과인 경우
                data = [results]
        else:
            # 이미 리스트 형식인 경우
            data = results
        
        # CSV 파일 작성
        if data:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    else:
        logger.error(f"지원하지 않는 형식: {format}")
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"결과 저장 완료: {output_path}")

def load_detections(det_path):
    """
    탐지 결과 파일 로드 (MOT Challenge 형식)
    
    Args:
        det_path: 탐지 결과 파일 경로
    
    Returns:
        detections: 프레임별 탐지 결과 딕셔너리
    """
    if not os.path.exists(det_path):
        logger.warning(f"탐지 결과 파일이 존재하지 않습니다: {det_path}")
        return {}
    
    detections = {}
    
    try:
        with open(det_path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                frame_id = int(data[0])
                obj_id = int(data[1])
                x, y, w, h = map(float, data[2:6])
                conf = float(data[6])
                
                if frame_id not in detections:
                    detections[frame_id] = []
                
                # (x, y, w, h) -> (x1, y1, x2, y2) 변환
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                detections[frame_id].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': 0  # MOT Challenge에서는 모두 사람 클래스
                })
    
    except Exception as e:
        logger.error(f"탐지 결과 파일 로드 중 오류 발생: {e}")
    
    return detections

def save_mot_results(results, output_path):
    """
    MOT Challenge 형식으로 추적 결과 저장
    
    Args:
        results: 추적 결과 리스트 (각 항목은 [frame_id, track_id, x1, y1, x2, y2, conf] 형식)
        output_path: 출력 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            # MOT Challenge 형식: frame_id, track_id, x, y, w, h, conf, -1, -1, -1
            frame_id, track_id, x1, y1, x2, y2, conf = result
            w, h = x2 - x1, y2 - y1
            f.write(f"{int(frame_id)},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")
    
    logger.info(f"MOT 형식 결과 저장 완료: {output_path}")