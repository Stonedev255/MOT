#!/bin/bash

# 모든 탐지기와 특징 추출기 조합 실행 스크립트

echo "===== 모든 탐지기와 특징 추출기 조합 실행 시작: $(date) ====="

# 탐지기 목록
detectors=("yolov5" "yolov8" "faster_rcnn" "fcos" "retinanet" "detr" "ssd" "mask_rcnn")

# 특징 추출기 목록
extractors=("resnet50" "swin" "convnext" "efficientnetv2")

# 데이터셋
dataset="MOT20"

# 각 탐지기와 특징 추출기 조합 실행
for detector in "${detectors[@]}"; do
    for extractor in "${extractors[@]}"; do
        echo "===== 조합: $detector + $extractor ====="
        echo "===== 스크립트 실행 시작 ====="
        
        python main.py \
            --detector "$detector" \
            --tracker "deepsort" \
            --extractor "$extractor" \
            --max-cosine-distance 0.2 \
            --max-iou-distance 0.7 \
            --max-age 30 \
            --n-init 3 \
            --nn-budget 100 \
            --dataset "$dataset" \
            --save-video \
            --save-txt \
            --output-format "json" \
            --output-dir "results" \
            --conf-thres 0.25 \
            --iou-thres 0.4
        
        echo "===== 조합: $detector + $extractor 완료 ====="
    done
done

echo "===== 모든 조합 실행 완료: $(date) ====="