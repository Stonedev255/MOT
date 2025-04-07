#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
평가 지표 계산 유틸리티 모듈
MOT 평가 지표(MOTA, MOTP, MT, ML 등) 계산 기능 제공
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def calculate_mot_metrics(tracking_results, gt_path, sequence_name):
    """
    MOT 평가 지표 계산
    
    Args:
        tracking_results: 추적 결과 리스트 (각 항목은 [frame_id, track_id, x1, y1, x2, y2, conf] 형식)
        gt_path: Ground Truth 파일 경로
        sequence_name: 시퀀스 이름
        
    Returns:
        metrics: 계산된 평가 지표 딕셔너리
    """
    try:
        # 평가를 위한 motmetrics 임포트
        import motmetrics as mm
        
        # 누산기 초기화
        acc = mm.MOTAccumulator(auto_id=True)
        
        # 추적 결과 프레임별로 정리
        track_frames = defaultdict(list)
        for result in tracking_results:
            frame_id, track_id, x1, y1, x2, y2, conf = result
            track_frames[int(frame_id)].append({
                'track_id': int(track_id),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(conf)
            })
        
        # GT 데이터 로드
        gt_frames = load_ground_truth(gt_path)
        
        # 모든 프레임 ID 추출
        all_frames = sorted(set(list(track_frames.keys()) + list(gt_frames.keys())))
        
        # 각 프레임별로 평가 수행
        for frame_id in all_frames:
            # 현재 프레임의 GT와 추적 결과 가져오기
            gt_dets = gt_frames.get(frame_id, [])
            track_dets = track_frames.get(frame_id, [])
            
            # GT와 추적 결과가 모두 있는 경우에만 평가 수행
            if gt_dets and track_dets:
                # GT 객체 ID와 바운딩 박스 추출
                gt_ids = [det['gt_id'] for det in gt_dets]
                gt_boxes = np.array([det['bbox'] for det in gt_dets])
                
                # 추적 객체 ID와 바운딩 박스 추출
                track_ids = [det['track_id'] for det in track_dets]
                track_boxes = np.array([det['bbox'] for det in track_dets])
                
                # IoU 계산
                distances = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
                
                # 누산기 업데이트
                acc.update(gt_ids, track_ids, distances)
        
        # 평가 지표 계산
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=[
            'mota', 'motp', 'num_switches',
            'mostly_tracked', 'mostly_lost',
            'num_fragmentations',
            'idf1', 'idp', 'idr',  # ID 관련 지표 추가
            'precision', 'recall',  # 정밀도, 재현율 추가
            'num_false_positives', 'num_misses',  # FP, FN 추가
        ], name=sequence_name)
        
        # 결과 추출
        mota = summary['mota'].iloc[0] * 100  # 백분율로 변환
        motp = (1 - summary['motp'].iloc[0]) * 100 if 'motp' in summary else 0  # 백분율로 변환
        num_switches = summary['num_switches'].iloc[0]
        mostly_tracked = summary['mostly_tracked'].iloc[0]
        mostly_lost = summary['mostly_lost'].iloc[0]
        num_fragmentations = summary['num_fragmentations'].iloc[0]
        
        # 추가 지표
        idf1 = summary['idf1'].iloc[0] * 100  # 백분율로 변환
        idp = summary['idp'].iloc[0] * 100  # ID 정밀도
        idr = summary['idr'].iloc[0] * 100  # ID 재현율
        precision = summary['precision'].iloc[0] * 100  # 정밀도 
        recall = summary['recall'].iloc[0] * 100  # 재현율
        fp = summary['num_false_positives'].iloc[0]  # 오검지 (False Positives)
        fn = summary['num_misses'].iloc[0]  # 미검지 (False Negatives)
        
        # 결과 반환
        metrics = {
            'mota': float(mota),
            'motp': float(motp),
            'num_switches': int(num_switches),
            'mostly_tracked': int(mostly_tracked),
            'mostly_lost': int(mostly_lost),
            'num_fragmentations': int(num_fragmentations),
            'idf1': float(idf1),
            'idp': float(idp),
            'idr': float(idr),
            'precision': float(precision),
            'recall': float(recall),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"평가 지표 계산 중 오류 발생: {e}")
        
        # 오류 발생 시 기본값 반환
        metrics = {
            'mota': 0.0,
            'motp': 0.0,
            'num_switches': 0,
            'mostly_tracked': 0,
            'mostly_lost': 0,
            'num_fragmentations': 0,
            'idf1': 0.0,
            'idp': 0.0,
            'idr': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'fp': 0,
            'fn': 0
        }
        
        return metrics

def load_ground_truth(gt_path):
    """
    Ground Truth 파일 로드 (MOT Challenge 형식)
    
    Args:
        gt_path: Ground Truth 파일 경로
        
    Returns:
        gt_frames: 프레임별 GT 딕셔너리
    """
    # Ground Truth 데이터가 없는 경우
    if not gt_path or not os.path.exists(gt_path):
        logger.warning(f"Ground Truth 파일이 존재하지 않습니다: {gt_path}")
        return {}
    
    gt_frames = defaultdict(list)
    
    try:
        # GT 파일 로드
        with open(gt_path, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                frame_id = int(data[0])
                obj_id = int(data[1])
                x, y, w, h = map(float, data[2:6])
                visibility = float(data[8])
                
                # 유효한 GT만 사용 (가시성이 일정 이상이고, 클래스가 person(1)인 경우)
                if int(data[6]) == 1 and int(data[7]) == 1 and visibility >= 0.25:
                    # (x, y, w, h) -> (x1, y1, x2, y2) 변환
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    gt_frames[frame_id].append({
                        'gt_id': obj_id,
                        'bbox': [x1, y1, x2, y2],
                        'visibility': visibility
                    })
    
    except Exception as e:
        logger.error(f"Ground Truth 파일 로드 중 오류 발생: {e}")
    
    return gt_frames

def calculate_average_metrics(metrics_list):
    """
    여러 시퀀스의 평가 지표 평균 계산
    
    Args:
        metrics_list: 시퀀스별 평가 지표 리스트
        
    Returns:
        avg_metrics: 평균 평가 지표 딕셔너리
    """
    if not metrics_list:
        return {
            'mota': 0.0,
            'motp': 0.0,
            'num_switches': 0,
            'mostly_tracked': 0,
            'mostly_lost': 0,
            'num_fragmentations': 0,
            'fps': 0.0
        }
    
    # 키별로 값 합산
    summed_metrics = defaultdict(float)
    count_metrics = defaultdict(int)
    
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key in ['mota', 'motp', 'fps']:  # 평균을 계산할 지표
                summed_metrics[key] += value
                count_metrics[key] += 1
            elif key in ['num_switches', 'mostly_tracked', 'mostly_lost', 'num_fragmentations']:  # 합산할 지표
                summed_metrics[key] += value
                count_metrics[key] = 1  # 합산 지표는 항상 1로 설정
    
    # 평균 계산
    avg_metrics = {}
    for key in summed_metrics:
        if count_metrics[key] > 0:
            if key in ['mota', 'motp', 'fps']:  # 평균을 계산할 지표
                avg_metrics[key] = summed_metrics[key] / count_metrics[key]
            else:  # 합산할 지표
                avg_metrics[key] = summed_metrics[key]
    
    return avg_metrics