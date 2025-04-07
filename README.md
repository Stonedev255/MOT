아래는 지금까지의 디렉토리 구조, 진행 방법, 사용 방법, 프로젝트 세부 내용을 기반으로 작성된 `README.md`입니다. GitHub 관련 내용은 제외하고, 프로젝트의 핵심 정보만 포함했습니다.

---

# MOT Performance Evaluation

이 프로젝트는 다양한 객체 탐지 모델과 추적 알고리즘(DeepSORT)을 결합하여 MOT(Multi-Object Tracking) 데이터셋(MOT16, MOT20)에 대한 성능을 평가합니다. YOLO 시리즈(YOLOv5, YOLOv7, YOLOv8), Faster R-CNN, FCOS, RetinaNet, DETR, SSD, Mask R-CNN 등의 탐지 모델을 지원하며, 다양한 특징 추출기(ResNet50, SWIN Transformer, ConvNeXt, EfficientNetV2)를 활용하여 추적 성능을 비교합니다.

---

## 디렉토리 구조

```
MOT/
├── README.md              # 프로젝트 설명 문서
├── configs/               # 설정 파일
│   ├── datasets/          # 데이터셋별 설정
│   │   ├── mot16.yaml
│   │   └── mot20.yaml
│   └── default.yaml       # 기본 설정 파일
├── detectors/             # 객체 탐지기 모듈
│   ├── __init__.py
│   ├── detector_factory.py
│   ├── faster_rcnn_detector.py
│   ├── fcos_detector.py
│   ├── retinanet_detector.py
│   ├── yolov5_detector.py
│   ├── yolov7_detector.py
│   ├── yolov8_detector.py
│   ├── detr_detector.py
│   ├── ssd_detector.py
│   └── mask_rcnn_detector.py
├── trackers/              # 추적기 모듈
│   ├── __init__.py
│   ├── tracker_factory.py
│   └── deep_sort/         # DeepSORT 구현
│       ├── deep_sort.py
│       ├── compat_extractor.py
│       └── core/          # DeepSORT 핵심 로직
├── models/                # 사전 학습된 모델 파일
│   ├── resnet50_extractor_pytorch.pth
│   ├── swin_extractor_pytorch.pth
│   ├── convnext_extractor_pytorch.pth
│   ├── efficientnetv2_extractor_pytorch.pth
│   └── object_detector_models/
│       ├── yolov5x.pt
│       ├── yolov7.pt
│       └── yolov8x.pt
├── utils/                 # 유틸리티 모듈
│   ├── __init__.py
│   ├── io_utils.py        # 데이터 로드 및 저장
│   ├── metrics.py         # 성능 평가 지표 계산
│   └── visualization.py   # 시각화 기능
├── main.py                # 메인 실행 스크립트
├── run_all_combinations.sh # 모든 탐지기+추출기 조합 실행
├── run_all_detectors.sh   # 모든 탐지기 실행
├── run_all_models.sh      # 모든 특징 추출기 실행
├── compare_detectors.py   # 탐지기별 성능 비교
├── compare_results.py     # 추출기별 성능 비교
├── requirements.txt       # 의존성 목록
└── results/               # 실행 결과 저장 디렉토리
    ├── mot20/
    │   ├── yolov5/
    │   │   ├── resnet50/
    │   │   │   ├── metrics/
    │   │   │   ├── tracks/
    │   │   │   └── visualizations/
    │   │   └── ...
    │   └── ...
```

---

## 프로젝트 세부 내용

### 목적
- MOT16 및 MOT20 데이터셋에서 다양한 객체 탐지 모델과 DeepSORT 추적 알고리즘의 성능을 평가.
- 탐지 모델과 특징 추출기의 조합에 따른 추적 성능(MOTA, MOTP, IDF1 등) 비교.

### 지원 탐지 모델
- **YOLO 시리즈**: YOLOv5, YOLOv7, YOLOv8
- **torchvision 기반**: Faster R-CNN, FCOS, RetinaNet, DETR, SSD, Mask R-CNN

### 지원 특징 추출기
- ResNet50
- SWIN Transformer
- ConvNeXt
- EfficientNetV2

### 평가 지표
- **MOTA**: 다중 객체 추적 정확도
- **MOTP**: 다중 객체 추적 정밀도
- **IDF1**: 식별 F1 스코어
- **Precision**: 정밀도
- **Recall**: 재현율
- **Num Switches**: ID 스위치 수
- **Mostly Tracked (MT)**: 대부분 추적된 객체 수
- **Mostly Lost (ML)**: 대부분 손실된 객체 수
- **Num Fragmentations**: 단절 수
- **FP**: 오검지 수
- **FN**: 미검지 수
- **FPS**: 초당 프레임 처리 속도

---

## 진행 방법

1. **환경 설정**:
   - 필요한 Python 패키지 설치:
     ```bash
     pip install -r requirements.txt
     ```
   - 의존성: `torch`, `torchvision`, `ultralytics`, `motmetrics`, `tabulate`, `pandas`, `matplotlib`, `tqdm`

2. **데이터셋 준비**:
   - MOT16 및 MOT20 데이터셋을 `/mnt/sda1/yrok/data/` 또는 사용자 지정 경로에 다운로드.
   - `configs/default.yaml`에서 `data_dir` 경로를 설정:
     ```yaml
     data_dir: /path/to/your/data
     ```

3. **실행 스크립트 준비**:
   - 각 스크립트에 실행 권한 부여:
     ```bash
     chmod +x run_all_combinations.sh run_all_detectors.sh run_all_models.sh
     ```

4. **실행**:
   - 모든 탐지기와 특징 추출기 조합 실행:
     ```bash
     ./run_all_combinations.sh
     ```
   - 특정 탐지기로 모든 특징 추출기 실행 (예: YOLOv8):
     ```bash
     ./run_all_models.sh
     ```
   - 모든 탐지기 실행 (특정 추출기 고정, 예: ResNet50):
     ```bash
     ./run_all_detectors.sh
     ```

5. **결과 분석**:
   - 탐지기별 비교:
     ```bash
     python compare_detectors.py --dataset mot20 --extractor resnet50
     ```
   - 특징 추출기별 비교:
     ```bash
     python compare_results.py --dataset mot20 --detector yolov8
     ```

---

## 사용 방법

### 개별 실행
- `main.py`를 사용하여 특정 탐지기와 추출기로 MOT 평가:
  ```bash
  python main.py \
      --detector yolov8 \
      --dataset MOT20 \
      --extractor resnet50 \
      --conf-thres 0.25 \
      --iou-thres 0.4 \
      --save-video
  ```

### 모든 조합 실행
- 모든 탐지기와 특징 추출기 조합을 테스트:
  ```bash
  ./run_all_combinations.sh
  ```
  - 출력: `results/mot20/[detector]/[extractor]/`에 저장.

### 결과 비교
- 탐지기별 성능 비교:
  ```bash
  python compare_detectors.py --dataset mot20 --extractor resnet50 --output detector_comparison_results
  ```
  - 출력: `detector_comparison_results/`에 그래프와 CSV 파일 생성.
- 특징 추출기별 성능 비교:
  ```bash
  python compare_results.py --dataset mot20 --detector yolov8 --output comparison_results_yolov8
  ```
  - 출력: `comparison_results_yolov8/`에 그래프 생성.

---

## 프로젝트 실행 예시

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 단일 실행
```bash
python main.py --detector faster_rcnn --dataset MOT20 --extractor swin --save-video
```

### 3. 모든 탐지기 실행
```bash
./run_all_detectors.sh
```

### 4. 결과 확인
- 결과 디렉토리 구조:
  ```
  results/mot20/faster_rcnn/swin/
  ├── metrics/           # 평가 지표 (JSON)
  ├── tracks/            # 추적 결과 (TXT)
  └── visualizations/    # 시각화 비디오 (MP4)
  ```

---

## 참고
- **데이터셋**: MOT16 및 MOT20는 [MOT Challenge](https://motchallenge.net/)에서 다운로드 가능.
- **의존성**: `requirements.txt`에 명시된 패키지가 설치되어 있어야 함.
- **문제 해결**: 실행 중 오류 발생 시, 로그 파일(`results/` 내 생성)을 확인하거나 콘솔 출력을 점검.

---

이 `README.md`는 프로젝트의 핵심 내용을 간결하게 정리했으며, 사용자가 쉽게 접근하고 실행할 수 있도록 구성했습니다. 추가로 포함하고 싶은 내용이나 수정할 부분이 있다면 말씀해주세요!