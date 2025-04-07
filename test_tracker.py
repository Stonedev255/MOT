from detectors.detector_factory import create_detector
from trackers.tracker_factory import create_tracker
import cv2
import logging

logging.basicConfig(level=logging.INFO)
detector = create_detector('yolov8', conf_thres=0.5, iou_thres=0.5)
tracker = create_tracker('deepsort')

# 테스트 이미지 로드
img_path = 'yolov5/data/images/bus.jpg'
image = cv2.imread(img_path)

if image is None:
    print(f"이미지를 로드할 수 없습니다: {img_path}")
else:
    # 객체 탐지
    detections = detector.detect(image)
    print(f"탐지된 객체 수: {len(detections)}")
    
    # 객체 추적
    tracks = tracker.update(detections, image)
    print(f"추적된 객체 수: {len(tracks)}")
    for i, track in enumerate(tracks[:5]):  # 처음 5개만 출력
        print(f"추적 객체 {i+1}: {track}")
