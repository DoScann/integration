'''
Python 3.11
opencv-python 4.9.x
'''

import os
import cv2
import numpy as np
import ctypes
from ctypes import wintypes


# 바탕화면 경로 가져오기 (Windows 전용)
def get_desktop_path():
    buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, 0, None, 0, buf)
    return buf.value


# 한글 경로에서도 안전하게 이미지를 읽기 위한 래퍼
def imread_unicode(path):
    if not os.path.isfile(path):
        print(f"파일 없음: {path}")
        return None

    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"이미지 디코딩 실패: {path}")
        return img
    except Exception as e:
        print(f"imread 실패: {path} ({e})")
        return None


# 한글 경로에서도 안전하게 이미지를 저장하기 위한 래퍼
def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]  # .png, .jpg 등
    if ext == "":
        ext = ".png"  # 확장자 없으면 png로 가정

    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            print(f"이미지 인코딩 실패: {path}")
            return False

        buf.tofile(path)
        return True
    except Exception as e:
        print(f"imwrite 실패: {path} ({e})")
        return False


DESKTOP_PATH = get_desktop_path()
input_folder = os.path.join(DESKTOP_PATH, "input")
output_folder = os.path.join(DESKTOP_PATH, "output_simple")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not files:
    print(f"input 폴더 경로: {input_folder}")
    print("여기에 이미지 넣고 다시 실행하면 됨.")
    raise SystemExit

# 업스케일 배율
scale = 2

# 보간법 (문서용으로 깔끔한 편)
interpolation = cv2.INTER_LANCZOS4

for file in files:
    in_path = os.path.join(input_folder, file)
    out_path = os.path.join(
        output_folder,
        f"{os.path.splitext(file)[0]}_x{scale}.png"
    )

    print(f"처리 중: {in_path!r}")

    img = imread_unicode(in_path)

    if img is None:
        print(f"로드 실패: {file}")
        continue

    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=interpolation)

    ok = imwrite_unicode(out_path, upscaled)
    if ok:
        print(f"저장 완료: {out_path}")
    else:
        print(f"저장 실패: {out_path}")

print("모든 업스케일 완료")
