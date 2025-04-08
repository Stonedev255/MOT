#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
평가 지표 계산 유틸리티 모듈
TrackEval을 사용해 MOT 평가 지표(HOTA, MOTA, IDF1 등) 계산 기능 제공
"""

import os
import sys
import numpy as np
import logging
from collections import defaultdict
from trackeval import Evaluator, datasets, metrics

logger = logging.getLogger(__name__)

def calculate_mot_metrics(tracking_results, gt_path, sequence_name):
    """
    TrackEval을 사용해 MOT 평가 지표 계산
    
    Args:
        tracking_results: 추적 결과 리스트 (각 항목은 [frame_id, track_id, x1, y1, x2, y2, conf] 형식)
        gt_path: Ground Truth 파일 경로
        sequence_name: 시퀀스 이름
        
    Returns:
        metric_results: 계산된 평가 지표 딕셔너리
    """
    print(f"DEBUG: calculate_mot_metrics 호출됨 - sequence_name: {sequence_name}")

    try:
        import shutil
        import tempfile
        from trackeval import metrics as track_metrics
        
        # 원본 시퀀스 폴더
        seq_folder = os.path.dirname(os.path.dirname(gt_path))  # /mnt/sda1/yrok/data/MOT20/train/MOT20-01
        
        # 임시 작업 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        print(f"DEBUG: 임시 디렉토리 생성됨: {temp_dir}")
        
        try:
            # TrackEval 설정
            config = {
                'TRACKERS_TO_EVAL': [sequence_name],
                'GT_FOLDER': temp_dir,  # 임시 디렉토리를 GT_FOLDER로 사용
                'TRACKERS_FOLDER': os.path.join(temp_dir, 'trackers'),
                'OUTPUT_FOLDER': None,
                'BENCHMARK': 'MOT20',
                'SPLIT_TO_EVAL': 'train',
                'DO_PREPROC': False,
                'METRICS': ['HOTA', 'CLEAR', 'Identity'],  # 이 설정은 평가하지 않을 때만 사용됨
            }
            print(f"DEBUG: config: {config}")
            
            # 필요한 디렉토리 구조 생성
            # 1. 시퀀스 디렉토리
            seq_temp_dir = os.path.join(temp_dir, sequence_name)
            os.makedirs(seq_temp_dir, exist_ok=True)
            
            # 2. GT 디렉토리
            gt_temp_dir = os.path.join(seq_temp_dir, 'gt')
            os.makedirs(gt_temp_dir, exist_ok=True)
            
            # 3. 트래커 디렉토리
            tracker_dir = os.path.join(config['TRACKERS_FOLDER'], sequence_name, 'data')
            os.makedirs(tracker_dir, exist_ok=True)
            
            # 4. seqinfo.ini 복사
            seqinfo_src = os.path.join(seq_folder, 'seqinfo.ini')
            seqinfo_dest = os.path.join(seq_temp_dir, 'seqinfo.ini')
            shutil.copy(seqinfo_src, seqinfo_dest)
            print(f"DEBUG: seqinfo.ini 복사됨: {seqinfo_src} -> {seqinfo_dest}")
            
            # 5. GT 파일 복사
            gt_src = gt_path
            gt_dest = os.path.join(gt_temp_dir, 'gt.txt')
            shutil.copy(gt_src, gt_dest)
            print(f"DEBUG: GT 파일 복사됨: {gt_src} -> {gt_dest}")
            
            # 6. 추적 결과 파일 생성
            tracker_file = os.path.join(tracker_dir, f"{sequence_name}.txt")
            with open(tracker_file, 'w') as f:
                for result in tracking_results:
                    frame_id, track_id, x1, y1, x2, y2, conf = result
                    w, h = x2 - x1, y2 - y1
                    f.write(f"{int(frame_id)},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")
            print(f"DEBUG: 추적 파일 생성됨: {tracker_file}")
            
            # TrackEval 데이터셋 초기화
            dataset_config = {
                'GT_FOLDER': config['GT_FOLDER'],
                'TRACKERS_FOLDER': config['TRACKERS_FOLDER'],
                'BENCHMARK': config['BENCHMARK'],
                'SPLIT_TO_EVAL': config['SPLIT_TO_EVAL'],
                'DO_PREPROC': config['DO_PREPROC'],
                'TRACKER_SUB_FOLDER': 'data',
                'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',  # 상대 경로 사용
                'CLASSES_TO_EVAL': ['pedestrian'],
                'SEQ_INFO': {sequence_name: None},
                'SKIP_SPLIT_FOL': True,
            }
            print(f"DEBUG: dataset_config: {dataset_config}")
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@000000000000000")
            
            # 데이터셋 생성
            dataset = datasets.MotChallenge2DBox(dataset_config)
            print(f"DEBUG: dataset 생성됨: {dataset}")
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@1111111111111")
            
            # 평가기 생성 및 실행
            evaluator = Evaluator(config)
            print(f"DEBUG: evaluator 생성됨: {evaluator}")
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22222222222222")
            
            # 평가할 메트릭 객체 생성
            metric_list = []
            for metric_name in ['HOTA', 'CLEAR', 'Identity']:
                if metric_name == 'HOTA':
                    metric_list.append(track_metrics.HOTA())
                elif metric_name == 'CLEAR':
                    metric_list.append(track_metrics.CLEAR())
                elif metric_name == 'Identity':
                    metric_list.append(track_metrics.Identity())
            
            # 평가 실행 - 데이터셋 객체를 리스트에 담아 전달
            # 결과 처리 부분 수정
            metrics_list = evaluator.evaluate([dataset], metric_list)
            print(f"DEBUG: TrackEval 결과 구조: {metrics_list}")
            logger.info(f"TrackEval 결과 구조: {metrics_list}")

            # 결과가 튜플인 경우 첫 번째 요소를 추출 (TrackEval 최신 버전에서는 튜플을 반환)
            if isinstance(metrics_list, tuple):
                metrics_data = metrics_list[0]  # 첫 번째 요소는 실제 지표 데이터
            else:
                metrics_data = metrics_list  # 이전 버전 호환성 유지
                
            # 이제 metrics_data를 사용하여 결과 처리 계속
            if isinstance(metrics_data, dict):
                print("DEBUG: metrics_data는 딕셔너리입니다")
                tracker_results = metrics_data.get('MotChallenge2DBox', {}).get('MOT20-01', {}).get('MOT20-01', {}).get('pedestrian', {})
            else:
                print("DEBUG: metrics_data는 딕셔너리가 아닙니다")
                raise ValueError(f"예상치 못한 TrackEval 결과 형식: {type(metrics_data)}")

            # 지표 추출
            print(f"DEBUG: tracker_results: {tracker_results}")
            hota_metrics = tracker_results.get('HOTA', {})
            clear_metrics = tracker_results.get('CLEAR', {})
            id_metrics = tracker_results.get('Identity', {})

            # 지표 정리
            metric_results = {
                'mota': float(clear_metrics.get('MOTA', 0) * 100),
                'motp': float(clear_metrics.get('MOTP', 0) * 100),
                'num_switches': int(id_metrics.get('IDSW', 0)),
                'mostly_tracked': int(clear_metrics.get('MT', 0)),
                'mostly_lost': int(clear_metrics.get('ML', 0)),
                'num_fragmentations': int(clear_metrics.get('Frag', 0)),
                'idf1': float(id_metrics.get('IDF1', 0) * 100),
                'precision': float(clear_metrics.get('CLR_Pr', 0) * 100),
                'recall': float(clear_metrics.get('CLR_Re', 0) * 100),
                'fp': int(clear_metrics.get('CLR_FP', 0)),
                'fn': int(clear_metrics.get('CLR_FN', 0)),
                'hota': float(hota_metrics.get('HOTA(0)', 0) * 100),
}
            
            print(f"DEBUG: 계산된 지표: {metric_results}")
            logger.info(f"계산된 지표: {metric_results}")
            
            return metric_results
            
        finally:
            # 임시 디렉토리 정리
            print(f"DEBUG: 임시 디렉토리 정리: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"평가 지표 계산 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()  # 상세 오류 메시지 출력
        metric_results = {
            'mota': 0.0,
            'motp': 0.0,
            'num_switches': 0,
            'mostly_tracked': 0,
            'mostly_lost': 0,
            'num_fragmentations': 0,
            'idf1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'fp': 0,
            'fn': 0,
            'hota': 0.0,
        }
        return metric_results