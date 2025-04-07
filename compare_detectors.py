#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
다양한 객체 탐지기를 비교하는 스크립트
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='다양한 객체 탐지기 비교')
    parser.add_argument('--dataset', type=str, default='mot20',
                       help='데이터셋 이름 (mot16 또는 mot20)')
    parser.add_argument('--detectors', type=str, nargs='+', 
                    default=['yolov5', 'yolov7', 'yolov8', 'faster_rcnn', 'fcos', 'retinanet', 'detr', 'ssd', 'mask_rcnn'],
                    help='비교할 탐지기 목록')
    parser.add_argument('--extractor', type=str, default='resnet50',
                       help='사용할 특징 추출기')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['MOTA', 'IDF1', 'Precision', 'Recall', 'ID Switches', 'FPS'],
                       help='비교할 지표 목록')
    parser.add_argument('--output', type=str, default='detector_comparison_results',
                       help='결과 저장 디렉토리')
    return parser.parse_args()

def load_results(dataset, detector, extractor):
    result_path = f'results/{dataset}/{detector}/{extractor}/metrics/summary.json'
    if not os.path.exists(result_path):
        print(f"결과 파일이 없습니다: {result_path}")
        return None
    
    with open(result_path, 'r') as f:
        return json.load(f)

def compare_detectors(args):
    results = {}
    
    # 각 탐지기별 결과 로드
    for detector in args.detectors:
        results[detector] = load_results(args.dataset, detector, args.extractor)
    
    # 결과가 없는 경우 제외
    results = {k: v for k, v in results.items() if v is not None}
    
    if not results:
        print("비교할 결과가 없습니다.")
        return
    
    # 표 형식으로 비교 결과 출력
    headers = ['시퀀스'] + args.metrics
    
    # 각 시퀀스별로 비교
    sequences = list(next(iter(results.values())).keys())
    
    for sequence in sequences:
        print(f"\n=== 시퀀스: {sequence} ===")
        
        table_data = []
        for detector, result in results.items():
            if sequence in result:
                row = [detector]
                for metric in args.metrics:
                    row.append(result[sequence].get(metric, 'N/A'))
                table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 각 지표별 평균 비교
    print("\n=== 평균 성능 비교 ===")
    avg_table = []
    
    for detector, result in results.items():
        row = [detector]
        for metric in args.metrics:
            values = [seq_result.get(metric, 0) for seq_result in result.values() 
                     if isinstance(seq_result.get(metric, 0), (int, float))]
            if values:
                avg = sum(values) / len(values)
                row.append(f"{avg:.2f}")
            else:
                row.append('N/A')
        avg_table.append(row)
    
    print(tabulate(avg_table, headers=headers, tablefmt='grid'))
    
    # 결과를 CSV 파일로 저장
    os.makedirs(args.output, exist_ok=True)
    df = pd.DataFrame(avg_table, columns=headers)
    csv_path = os.path.join(args.output, f'detector_comparison_{args.dataset}_{args.extractor}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n결과가 CSV 파일로 저장되었습니다: {csv_path}")
    
    # 그래프로 시각화
    for metric in args.metrics:
        plt.figure(figsize=(10, 6))
        
        detector_names = []
        metric_values = []
        
        for detector, result in results.items():
            detector_names.append(detector)
            values = [seq_result.get(metric, 0) for seq_result in result.values() 
                     if isinstance(seq_result.get(metric, 0), (int, float))]
            if values:
                avg = sum(values) / len(values)
                metric_values.append(avg)
            else:
                metric_values.append(0)
        
        # 막대 그래프 생성
        bars = plt.bar(detector_names, metric_values)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.5, 
                    f"{value:.2f}", ha='center', va='bottom')
        
        plt.title(f'{metric} 비교 ({args.dataset.upper()}, {args.extractor})')
        plt.xlabel('객체 탐지기')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 그래프 저장
        output_path = os.path.join(args.output, f'detector_comparison_{args.dataset}_{args.extractor}_{metric}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"{output_path} 저장 완료")
    
    # 모든 지표를 한 번에 비교하는 막대 그래프 생성
    plt.figure(figsize=(15, 8))
    
    # 데이터 준비
    x = np.arange(len(args.metrics))
    width = 0.8 / len(results)
    
    # 각 탐지기별로 막대 그래프 추가
    for i, (detector, result) in enumerate(results.items()):
        # 각 지표별 평균 계산
        avgs = []
        for metric in args.metrics:
            values = [seq_result.get(metric, 0) for seq_result in result.values() 
                     if isinstance(seq_result.get(metric, 0), (int, float))]
            if values:
                avg = sum(values) / len(values)
                # ID Switches와 같은 값은 적을수록 좋으므로 정규화
                if metric in ['ID Switches', 'ML', 'FN', 'FP']:
                    # 값이 매우 크면 로그 스케일 사용
                    if avg > 1000:
                        avg = np.log10(avg)
                        avgs.append(avg)
                    else:
                        # 이 경우 값이 작을수록 좋으므로 표시하지만 별도 처리 필요
                        avgs.append(avg)
                else:
                    avgs.append(avg)
            else:
                avgs.append(0)
        
        # 막대 그래프 추가
        plt.bar(x + i*width - 0.4 + width/2, avgs, width, label=detector)
    
    plt.xlabel('평가 지표')
    plt.ylabel('평균 값')
    plt.title(f'객체 탐지기별 성능 비교 ({args.dataset.upper()}, {args.extractor})')
    plt.xticks(x, args.metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 종합 그래프 저장
    output_path = os.path.join(args.output, f'detector_comparison_{args.dataset}_{args.extractor}_all_metrics.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"{output_path} 저장 완료")

if __name__ == "__main__":
    args = parse_args()
    compare_detectors(args)