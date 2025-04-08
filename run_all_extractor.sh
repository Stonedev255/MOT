#!/bin/bash
# 모든 특징 추출기를 실행하고 결과를 비교하는 스크립트

# 설정
DATASET="MOT20"        # 데이터셋 (MOT16 또는 MOT20)
DETECTOR="yolov8"      # 탐지기 (CenterNet 대신 yolov8으로 대체)
CONF_THRES="0.25"      # 신뢰도 임계값
IOU_THRES="0.4"        # IOU 임계값
SAVE_VIDEO="--save-video"  # 비디오 저장 여부 (빈 문자열로 변경하면 비디오 저장 안 함)

# 결과 디렉토리
RESULTS_DIR="comparison_results_${DETECTOR}"
mkdir -p $RESULTS_DIR

# 시작 시간 기록
echo "===== 모든 특징 추출기 실행 시작: $(date) ====="
START_TIME=$(date +%s)

# 특징 추출기 목록
EXTRACTORS=("resnet50" "swin" "convnext" "efficientnetv2")

# 각 특징 추출기 실행
for EXTRACTOR in "${EXTRACTORS[@]}"; do
    echo -e "\n===== $EXTRACTOR 특징 추출기 실행 ====="
    python3 main.py \
        --detector "$DETECTOR" \
        --dataset "$DATASET" \
        --conf-thres "$CONF_THRES" \
        --iou-thres "$IOU_THRES" \
        --tracker "deepsort" \
        --extractor "$EXTRACTOR" \
        --max-cosine-distance 0.2 \
        --max-iou-distance 0.7 \
        --max-age 30 \
        --n-init 3 \
        --nn-budget 100 \
        $SAVE_VIDEO
done

# 완료 시간 기록 및 소요 시간 계산
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo -e "\n===== 모든 특징 추출기 실행 완료: $(date) ====="
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"

# 결과 비교
echo -e "\n===== 특징 추출기 결과 비교 ====="
if [ -f "compare_results.py" ]; then
    python compare_results.py \
        --dataset "${DATASET,,}" \
        --detector "$DETECTOR" \
        --output "$RESULTS_DIR"
    echo "결과 비교 완료. 비교 그래프가 $RESULTS_DIR 디렉토리에 저장되었습니다."
else
    echo "compare_results.py 파일이 없습니다. 결과 비교를 건너뜁니다."
fi

echo -e "\n===== 모든 작업 완료 ====="