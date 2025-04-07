#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시각화 유틸리티 모듈
객체 탐지 및 추적 결과를 시각화하는 기능 제공
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# 고유한 색상 생성을 위한 함수
def create_unique_color_uchar(tag, hue_step=0.41):
    """
    고유한 색상 생성
    
    Args:
        tag: 색상을 생성할 태그 (숫자 ID)
        hue_step: 색상 간격
        
    Returns:
        color: (B, G, R) 형식의 색상 튜플
    """
    h = (tag * hue_step) % 1
    v = 1.0 - (tag * hue_step % 0.5)
    r, g, b = hsv_to_rgb(h, 1, v)
    return int(b * 255), int(g * 255), int(r * 255)

def hsv_to_rgb(h, s, v):
    """
    HSV 색상 공간을 RGB로 변환
    
    Args:
        h: 색조 (0~1)
        s: 채도 (0~1)
        v: 명도 (0~1)
        
    Returns:
        (r, g, b): RGB 색상 튜플 (0~1)
    """
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    i %= 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def visualize_tracking_results(image, tracks, frame_idx):
    """
    추적 결과 시각화
    
    Args:
        image: 원본 이미지 (BGR 형식)
        tracks: 추적 결과 리스트, 각 항목은 {'track_id': int, 'bbox': [x1, y1, x2, y2], 'confidence': float} 형식
        frame_idx: 프레임 인덱스
        
    Returns:
        image: 시각화된 이미지
    """
    # 이미지 복사
    vis_image = image.copy()
    
    # 프레임 인덱스 표시
    text = f'Frame: {frame_idx}'
    cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 추적 객체가 없는 경우
    if not tracks:
        return vis_image
    
    # 각 추적 객체 그리기
    for track in tracks:
        track_id = track['track_id']
        bbox = track['bbox']
        confidence = track.get('confidence', 1.0)
        
        # 박스 좌표
        x1, y1, x2, y2 = map(int, bbox)
        
        # 고유한 색상 생성
        color = create_unique_color_uchar(track_id)
        
        # 바운딩 박스 그리기
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # ID 및 신뢰도 표시
        text = f'ID: {track_id}'
        if confidence < 1.0:
            text += f' {confidence:.2f}'
        
        cv2.putText(vis_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_image

def plot_trajectories(tracks, output_path, image_shape=None):
    """
    추적 궤적 시각화
    
    Args:
        tracks: 추적 결과 리스트 (각 항목은 [frame_id, track_id, x1, y1, x2, y2, conf] 형식)
        output_path: 출력 이미지 경로
        image_shape: 원본 이미지 크기 (height, width)
    """
    # 트랙별로 궤적 구성
    trajectories = defaultdict(list)
    
    for track in tracks:
        frame_id, track_id, x1, y1, x2, y2, conf = track
        # 중심점 계산
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        trajectories[int(track_id)].append((int(frame_id), float(cx), float(cy)))
    
    # 그림 초기화
    plt.figure(figsize=(12, 8))
    
    # 이미지 크기가 제공된 경우 좌표축 설정
    if image_shape:
        height, width = image_shape
        plt.xlim(0, width)
        plt.ylim(height, 0)  # y축은 위아래 반전 (이미지 좌표계)
    
    # 각 트랙의 궤적 그리기
    for track_id, points in trajectories.items():
        # 포인트 정렬 (프레임 기준)
        points.sort(key=lambda x: x[0])
        
        # 좌표 추출
        frames, xs, ys = zip(*points)
        
        # 색상 생성
        color = hsv_to_rgb((track_id * 0.41) % 1, 1, 1)
        
        # 궤적 그리기
        plt.plot(xs, ys, '-', color=color, linewidth=1, alpha=0.7)
        plt.plot(xs, ys, 'o', color=color, markersize=3)
        
        # 시작점과 끝점 표시
        plt.plot(xs[0], ys[0], 'o', color=color, markersize=6)
        plt.plot(xs[-1], ys[-1], 's', color=color, markersize=6)
        
        # ID 표시
        plt.text(xs[-1], ys[-1], f'ID: {track_id}', color=color, fontsize=8)
    
    plt.title('Tracking Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    
    # 결과 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"궤적 시각화 저장 완료: {output_path}")

def create_comparison_visualizations(sequences, metrics_dict, output_path):
    """
    성능 비교 시각화
    
    Args:
        sequences: 시퀀스 이름 리스트
        metrics_dict: {detector_name: {sequence_name: metrics}} 형식의 성능 지표 딕셔너리
        output_path: 출력 이미지 경로
    """
    detectors = list(metrics_dict.keys())
    
    metrics_to_plot = [
        ('MOTA', 'mota', '%'),
        ('MOTP', 'motp', '%'),
        ('ID Switches', 'num_switches', ''),
        ('MT', 'mostly_tracked', ''),
        ('ML', 'mostly_lost', ''),
        ('FPS', 'fps', '')
    ]
    
    # 그리드 레이아웃 설정
    n_metrics = len(metrics_to_plot)
    n_rows = (n_metrics + 1) // 2
    n_cols = 2
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, (title, metric_key, unit) in enumerate(metrics_to_plot):
        plt.subplot(n_rows, n_cols, i + 1)
        
        bar_width = 0.2
        index = np.arange(len(sequences))
        
        for j, detector in enumerate(detectors):
            values = []
            for seq in sequences:
                if seq in metrics_dict[detector] and metric_key in metrics_dict[detector][seq]:
                    values.append(metrics_dict[detector][seq][metric_key])
                else:
                    values.append(0)
            
            plt.bar(index + j * bar_width, values, bar_width, label=detector)
        
        plt.xlabel('Sequence')
        plt.ylabel(f'{title} {unit}')
        plt.title(title)
        plt.xticks(index + bar_width * (len(detectors) - 1) / 2, sequences, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"성능 비교 시각화 저장 완료: {output_path}")