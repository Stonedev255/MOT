from utils.io_utils import load_dataset_sequences
import logging

logging.basicConfig(level=logging.INFO)
sequences = load_dataset_sequences('MOT16', '/mnt/sda1/yrok/data', 'MOT16-14')
print(f"로드된 시퀀스 수: {len(sequences)}")
if sequences:
    print(f"첫 번째 시퀀스 이름: {sequences[0]['name']}")
    print(f"이미지 수: {len(sequences[0]['images'])}")
