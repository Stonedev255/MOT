#!/usr/bin/env python
import os
import re
import sys

# TrackEval 소스 경로 확인
import trackeval
trackeval_path = os.path.dirname(trackeval.__file__)
print(f"TrackEval 경로: {trackeval_path}")

# 수정할 파일들 찾기
py_files = []
for root, dirs, files in os.walk(trackeval_path):
    for file in files:
        if file.endswith('.py'):
            py_files.append(os.path.join(root, file))

# 각 파일 수정
modified_files = []
for file_path in py_files:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # np.float -> np.float64 (정확한 단어 경계 확인)
    orig_content = content
    content = re.sub(r'np\.float\b(?!64)', r'np.float64', content)
    
    # np.int -> np.int64 (정확한 단어 경계 확인)
    content = re.sub(r'np\.int\b(?!64)', r'np.int64', content)
    
    # 잘못된 이중 대체 수정 (np.float6464 -> np.float64)
    content = re.sub(r'np\.float6464', r'np.float64', content)
    content = re.sub(r'np\.int6464', r'np.int64', content)
    
    # 변경되었으면 파일 업데이트
    if content != orig_content:
        with open(file_path, 'w') as f:
            f.write(content)
        modified_files.append(file_path)
        print(f"수정됨: {file_path}")

print(f"총 {len(modified_files)}개 파일이 수정되었습니다.")