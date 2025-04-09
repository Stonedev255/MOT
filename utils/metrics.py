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
import shutil
import tempfile
import io
from contextlib import redirect_stdout
from trackeval import Evaluator, datasets, metrics

logger = logging.getLogger(__name__)

def calculate_mot_metrics(tracking_results, gt_path, sequence_name, output_dir=None):
    """
    TrackEval을 사용해 MOT 평가 지표 계산
    
    Args:
        tracking_results: 추적 결과 리스트 (각 항목은 [frame_id, track_id, x1, y1, x2, y2, conf] 형식)
        gt_path: Ground Truth 파일 경로
        sequence_name: 시퀀스 이름
        output_dir: 텍스트 파일 저장 디렉토리 (기본값: None, 현재 디렉토리에 저장)
        
    Returns:
        metric_results: 계산된 평가 지표 딕셔너리
    """
    logger.debug(f"calculate_mot_metrics 호출됨 - sequence_name: {sequence_name}")

    temp_dir = tempfile.mkdtemp()
    logger.debug(f"임시 디렉토리 생성됨: {temp_dir}")
    
    try:
        seq_folder = os.path.dirname(os.path.dirname(gt_path))
        config = {
            'TRACKERS_TO_EVAL': [sequence_name],
            'GT_FOLDER': temp_dir,
            'TRACKERS_FOLDER': os.path.join(temp_dir, 'trackers'),
            'OUTPUT_FOLDER': None,
            'BENCHMARK': 'MOT20',
            'SPLIT_TO_EVAL': 'train',
            'DO_PREPROC': False,
        }
        logger.debug(f"config: {config}")
        
        seq_temp_dir = os.path.join(temp_dir, sequence_name)
        os.makedirs(seq_temp_dir, exist_ok=True)
        gt_temp_dir = os.path.join(seq_temp_dir, 'gt')
        os.makedirs(gt_temp_dir, exist_ok=True)
        tracker_dir = os.path.join(config['TRACKERS_FOLDER'], sequence_name, 'data')
        os.makedirs(tracker_dir, exist_ok=True)
        
        shutil.copy(os.path.join(seq_folder, 'seqinfo.ini'), os.path.join(seq_temp_dir, 'seqinfo.ini'))
        logger.debug(f"seqinfo.ini 복사됨")
        gt_dest = os.path.join(gt_temp_dir, 'gt.txt')
        shutil.copy(gt_path, gt_dest)
        logger.debug(f"GT 파일 복사됨")
        
        tracker_file = os.path.join(tracker_dir, f"{sequence_name}.txt")
        with open(tracker_file, 'w') as f:
            for result in tracking_results:
                frame_id, track_id, x1, y1, x2, y2, conf = result
                w, h = x2 - x1, y2 - y1
                f.write(f"{int(frame_id)},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")
        logger.debug(f"추적 파일 생성됨: {tracker_file}")
        
        dataset_config = {
            'GT_FOLDER': config['GT_FOLDER'],
            'TRACKERS_FOLDER': config['TRACKERS_FOLDER'],
            'BENCHMARK': config['BENCHMARK'],
            'SPLIT_TO_EVAL': config['SPLIT_TO_EVAL'],
            'DO_PREPROC': config['DO_PREPROC'],
            'TRACKER_SUB_FOLDER': 'data',
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
            # 'CLASSES_TO_EVAL' 제거: 모든 클래스를 평가하도록 설정
            'SEQ_INFO': {sequence_name: None},
            'SKIP_SPLIT_FOL': True,
        }
        logger.debug(f"dataset_config: {dataset_config}")
        
        dataset = datasets.MotChallenge2DBox(dataset_config)
        evaluator = Evaluator()
        metrics_list = [metrics.HOTA(), metrics.CLEAR(), metrics.Identity()]
        
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            results = evaluator.evaluate([dataset], metrics_list)
        output_text = output_buffer.getvalue()
        
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"trackeval_output_{sequence_name}.txt")
        with open(output_file, 'w') as f:
            f.write(output_text)
        logger.info(f"TrackEval 콘솔 출력 저장됨: {output_file}")
        
        if isinstance(results, tuple):
            metrics_data = results[0]
        else:
            metrics_data = results
        
        # 모든 클래스에 대한 결과를 저장하기 위해 클래스별로 처리
        metric_results = {}
        for class_name in metrics_data['MotChallenge2DBox'][sequence_name][sequence_name]:
            tracker_data = metrics_data['MotChallenge2DBox'][sequence_name][sequence_name][class_name]
            hota_metrics = tracker_data.get('HOTA', {})
            clear_metrics = tracker_data.get('CLEAR', {})
            id_metrics = tracker_data.get('Identity', {})
            count_metrics = tracker_data.get('Count', {})
            
            metric_results[class_name] = {
                'HOTA': float(np.mean(hota_metrics.get('HOTA', [0])) * 100),
                'DetA': float(np.mean(hota_metrics.get('DetA', [0])) * 100),
                'AssA': float(np.mean(hota_metrics.get('AssA', [0])) * 100),
                'DetRe': float(np.mean(hota_metrics.get('DetRe', [0])) * 100),
                'DetPr': float(np.mean(hota_metrics.get('DetPr', [0])) * 100),
                'AssRe': float(np.mean(hota_metrics.get('AssRe', [0])) * 100),
                'AssPr': float(np.mean(hota_metrics.get('AssPr', [0])) * 100),
                'LocA': float(np.mean(hota_metrics.get('LocA', [0])) * 100),
                'OWTA': float(np.mean(hota_metrics.get('OWTA', [0])) * 100),
                'HOTA(0)': float(hota_metrics.get('HOTA(0)', 0) * 100),
                'LocA(0)': float(hota_metrics.get('LocA(0)', 0) * 100),
                'HOTALocA(0)': float(hota_metrics.get('HOTALocA(0)', 0) * 100),
                'MOTA': float(clear_metrics.get('MOTA', 0) * 100),
                'MOTP': float(clear_metrics.get('MOTP', 0) * 100),
                'MODA': float(clear_metrics.get('MODA', 0) * 100),
                'CLR_Re': float(clear_metrics.get('CLR_Re', 0) * 100),
                'CLR_Pr': float(clear_metrics.get('CLR_Pr', 0) * 100),
                'MTR': float(clear_metrics.get('MTR', 0) * 100),
                'PTR': float(clear_metrics.get('PTR', 0) * 100),
                'MLR': float(clear_metrics.get('MLR', 0) * 100),
                'sMOTA': float(clear_metrics.get('sMOTA', 0) * 100),
                'CLR_TP': int(clear_metrics.get('CLR_TP', 0)),
                'CLR_FN': int(clear_metrics.get('CLR_FN', 0)),
                'CLR_FP': int(clear_metrics.get('CLR_FP', 0)),
                'IDSW': int(clear_metrics.get('IDSW', 0)),
                'MT': int(clear_metrics.get('MT', 0)),
                'PT': int(clear_metrics.get('PT', 0)),
                'ML': int(clear_metrics.get('ML', 0)),
                'Frag': int(clear_metrics.get('Frag', 0)),
                'IDF1': float(id_metrics.get('IDF1', 0) * 100),
                'IDR': float(id_metrics.get('IDR', 0) * 100),
                'IDP': float(id_metrics.get('IDP', 0) * 100),
                'IDTP': int(id_metrics.get('IDTP', 0)),
                'IDFN': int(id_metrics.get('IDFN', 0)),
                'IDFP': int(id_metrics.get('IDFP', 0)),
                'Dets': int(count_metrics.get('Dets', 0)),
                'GT_Dets': int(count_metrics.get('GT_Dets', 0)),
                'IDs': int(count_metrics.get('IDs', 0)),
                'GT_IDs': int(count_metrics.get('GT_IDs', 0))
            }
        
        logger.debug(f"계산된 지표: {metric_results}")
        return metric_results
        
    except Exception as e:
        logger.error(f"평가 지표 계산 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        metric_results = {
            'pedestrian': {
                'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0, 'DetRe': 0.0, 'DetPr': 0.0, 'AssRe': 0.0, 'AssPr': 0.0,
                'LocA': 0.0, 'OWTA': 0.0, 'HOTA(0)': 0.0, 'LocA(0)': 0.0, 'HOTALocA(0)': 0.0,
                'MOTA': 0.0, 'MOTP': 0.0, 'MODA': 0.0, 'CLR_Re': 0.0, 'CLR_Pr': 0.0, 'MTR': 0.0, 'PTR': 0.0,
                'MLR': 0.0, 'sMOTA': 0.0, 'CLR_TP': 0, 'CLR_FN': 0, 'CLR_FP': 0, 'IDSW': 0, 'MT': 0, 'PT': 0,
                'ML': 0, 'Frag': 0, 'IDF1': 0.0, 'IDR': 0.0, 'IDP': 0.0, 'IDTP': 0, 'IDFN': 0, 'IDFP': 0,
                'Dets': 0, 'GT_Dets': 0, 'IDs': 0, 'GT_IDs': 0
            }
        }
        return metric_results
    
    finally:
        logger.debug(f"임시 디렉토리 정리: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)