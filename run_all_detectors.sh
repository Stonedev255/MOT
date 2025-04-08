#!/bin/bash
# 모든 객체 탐지기를 실행하고 결과를 비교하는 스크립트

# 설정
DATASET="MOT20"         # 데이터셋 (MOT16 또는 MOT20)
EXTRACTOR="resnet50"    # 특징 추출기 (resnet50, swin, convnext, efficientnetv2)
CONF_THRES="0.25"       # 신뢰도 임계값
IOU_THRES="0.4"         # IOU 임계값
SAVE_VIDEO="--save-video"  # 비디오 저장 여부 (빈 문자열로 변경하면 비디오 저장 안 함)

# 결과 디렉토리
RESULTS_DIR="detector_comparison_results"
mkdir -p $RESULTS_DIR

# 시작 시간 기록
echo "===== 모든 탐지기 실행 시작: $(date) ====="
START_TIME=$(date +%s)

# 탐지기 목록 (CenterNet 제거, DETR, SSD, Mask R-CNN 추가)
DETECTORS=("yolov5" "yolov7" "yolov8" "faster_rcnn" "fcos" "retinanet" "detr" "ssd" "mask_rcnn")

# 각 탐지기 실행
for DETECTOR in "${DETECTORS[@]}"; do
    echo -e "\n===== $DETECTOR 실행 ====="
    python main.py \
        --detector "$DETECTOR" \
        --dataset "$DATASET" \
        --conf-thres "$CONF_THRES" \
        --iou-thres "$IOU_THRES" \
        $SAVE_VIDEO \
        --extractor "$EXTRACTOR"
done

# 완료 시간 기록 및 소요 시간 계산
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo -e "\n===== 모든 탐지기 실행 완료: $(date) ====="
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"

# 결과 비교
echo -e "\n===== 탐지기 결과 비교 ====="
if [ -f "compare_detectors.py" ]; then
    python compare_detectors.py \
        --dataset "${DATASET,,}" \
        --extractor "$EXTRACTOR" \
        --output "$RESULTS_DIR"
    echo "결과 비교 완료. 비교 그래프가 $RESULTS_DIR 디렉토리에 저장되었습니다."
else
    echo "compare_detectors.py 파일이 없습니다. 결과 비교를 건너뜁니다."
fi

echo -e "\n===== 모든 작업 완료 ====="