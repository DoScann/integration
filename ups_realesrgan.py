'''
Python 3.11
py_real_esrgan 0.3.1
torch 2.3.x (CPU)
pillow 10.x
tqdm 4.66.x
'''

import os
from PIL import Image
from py_real_esrgan.model import RealESRGAN
from tqdm import tqdm
import torch
import ctypes
from ctypes import wintypes


# ==========================================
# 윈도우 환경에서 "바탕화면" 경로 자동으로 가져오는 함수
# SHGetFolderPathW API 사용 (ctypes 기반)
# ==========================================
def get_desktop_path():
    buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, 0, None, 0, buf)
    return buf.value


# ==========================================
# 기본 경로 설정
# - input  : 업스케일링할 원본 이미지 폴더
# - output : 결과 이미지 저장 폴더
# - weights: RealESRGAN 모델 가중치 저장 폴더
# ==========================================
DESKTOP_PATH = get_desktop_path()
input_folder = os.path.join(DESKTOP_PATH, "input")
output_folder = os.path.join(DESKTOP_PATH, "output")
weights_folder = os.path.join(DESKTOP_PATH, "realesrgan_weights")

# 업스케일倍率 (예: x4)
scale = 4

# 사용 가능한 디바이스 확인 (GPU 있으면 CUDA, 없으면 CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 결과 저장용 폴더들 미리 생성
os.makedirs(output_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

# ==========================================
# input 폴더가 없으면 자동 생성 후 종료
# → 사용자 실수 방지: 빈 input 폴더에 이미지를 넣도록 안내
# ==========================================
if not os.path.isdir(input_folder):
    os.makedirs(input_folder, exist_ok=True)
    print(f"input 폴더 생성됨: {input_folder}")
    print("여기에 이미지 넣고 다시 실행하면 됨.")
    raise SystemExit


# ==========================================
# RealESRGAN 모델 로드
# - scale에 따라 x2, x4 가중치 파일 자동 선택
# - weights_path 위치에서 모델 가중치 읽어옴
# ==========================================
model = RealESRGAN(device, scale=scale)
weights_path = os.path.join(weights_folder, f"RealESRGAN_x{scale}.pth")
model.load_weights(weights_path)


# ==========================================
# 업스케일 수행
# - input 폴더 내 이미지 파일만 탐지
# - 예외 발생해도 전체 루프는 계속 실행
# ==========================================
exts = (".png", ".jpg", ".jpeg")

for file in tqdm(os.listdir(input_folder)):
    # 지원하는 확장자가 아니면 스킵
    if not file.lower().endswith(exts):
        continue

    # 입력 / 출력 파일 경로 구성
    in_path = os.path.join(input_folder, file)
    out_path = os.path.join(
        output_folder,
        f"{os.path.splitext(file)[0]}_x{scale}.png"
    )

    try:
        # Pillow로 이미지 로드 (RGB 통일)
        img = Image.open(in_path).convert("RGB")

        # RealESRGAN 모델로 업스케일 실행
        upscaled = model.predict(img)

        # 파일 저장
        upscaled.save(out_path)
        print(f"완료: {file} -> {out_path}")

    except Exception as e:
        # 처리 실패한 파일은 오류 표시하고 계속 진행
        print(f"실패: {file} ({e})")

print("완료")
