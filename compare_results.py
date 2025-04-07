#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모델별 결과 비교 스크립트
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description='모델별 결과 비교')
    parser.add_argument('--dataset', type=str, default='mot20',
                       help='데이터셋 이름 (mot16 또는 mot20)')
    parser.add_argument('--detector', type=str, default='yolov5',
                       help='객체 탐지기 이름 (yolov5, yolov7, yolov8)')
    parser.add_argument('--extractors', type=str, nargs='+', 
                       default=['resnet50', 'swin', 'convnext', 'efficientnetv2'],
                       help='비교할 특징 추출기 목록')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['MOTA', 'IDF1', 'Precision', 'Recall', 'ID Switches', 'FPS'],
                       help='비교할 지표 목록')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='결과 저장 디렉토리')
    return parser.parse_args()

def load_results(dataset, detector, extractor):
    result_path = f'results/{dataset}/{detector}/{extractor}/metrics/summary.json'
    if not os.path.exists(result_path):
        print(f"결과 파일이 없습니다: {result_path}")
        return None
    
    with open(result_path, 'r') as f:
        return json.load(f)

def compare_results(args):
    results = {}
    
    # 각 추출기별 결과 로드
    for extractor in args.extractors:
        results[extractor] = load_results(args.dataset, args.detector, extractor)
    
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
        for extractor, result in results.items():
            if sequence in result:
                row = [extractor]
                for metric in args.metrics:
                    row.append(result[sequence].get(metric, 'N/A'))
                table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 각 지표별 평균 비교
    print("\n=== 평균 성능 비교 ===")
    avg_table = []
    
    for extractor, result in results.items():
        row = [extractor]
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
    
    # 그래프로 시각화 (선택된 지표에 대해서만)
    os.makedirs(args.output, exist_ok=True)
    
    for metric in args.metrics:
        plt.figure(figsize=(12, 6))
        
        for extractor, result in results.items():
            seq_names = []
            metric_values = []
            
            for seq, seq_result in result.items():
                if isinstance(seq_result.get(metric, None), (int, float)):
                    seq_names.append(seq)
                    metric_values.append(seq_result.get(metric, 0))
            
            if metric_values:
                plt.plot(seq_names, metric_values, 'o-', label=extractor)
        
        plt.title(f'{metric} 비교')
        plt.xlabel('시퀀스')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 그래프 저장
        output_path = os.path.join(args.output, f'comparison_{metric}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"{output_path} 저장 완료")
    
    # 모든 지표를 한 번에 비교하는 막대 그래프 생성
    plt.figure(figsize=(15, 10))
    
    # 데이터 준비
    extractors = list(results.keys())
    x = np.arange(len(args.metrics))
    width = 0.8 / len(extractors)
    
    # 각 추출기별로 막대 그래프 추가
    for i, extractor in enumerate(extractors):
        result = results[extractor]
        
        # 각 지표별 평균 계산
        avgs = []
        for metric in args.metrics:
            values = [seq_result.get(metric, 0) for seq_result in result.values() 
                     if isinstance(seq_result.get(metric, 0), (int, float))]
            if values:
                avg = sum(values) / len(values)
                avgs.append(avg)
            else:
                avgs.append(0)
        
        # 막대 그래프 추가
        plt.bar(x + i*width - 0.4 + width/2, avgs, width, label=extractor)
    
    plt.xlabel('평가 지표')
    plt.ylabel('평균 값')
    plt.title('특징 추출기별 성능 비교')
    plt.xticks(x, args.metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 종합 그래프 저장
    output_path = os.path.join(args.output, 'comparison_all_metrics.png')
    plt.savefig(output_path)
    plt.close()
    print(f"{output_path} 저장 완료")

if __name__ == "__main__":
    args = parse_args()
    compare_results(args)