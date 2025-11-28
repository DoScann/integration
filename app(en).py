import io
import os
import base64
import json
import time
import traceback
import logging

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
import easyocr
from spellchecker import SpellChecker
from cv2 import dnn_superres

# 1. 설정 및 초기화
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

print("=== 서버 초기화 중 ===")

# [모드 1] 영어/스펠링 체크용 리소스
print("1. 영어 전용 OCR 로드 중...")
reader_en = easyocr.Reader(['en'], gpu=False) 
spell = SpellChecker()

# [모드 2] 한글/AI 고화질용 리소스
print("2. 한글 공용 OCR 로드 중...")
reader_ko = easyocr.Reader(['ko', 'en'], gpu=False) 

print("3. AI Super Resolution 모델 로드 중...")
sr = dnn_superres.DnnSuperResImpl_create()
model_path = os.path.join(BASE_DIR, "EDSR_x4.pb")
ai_loaded = False

if os.path.exists(model_path):
    try:
        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        ai_loaded = True
        print("   ✅ AI Model Loaded (EDSR_x4)")
    except Exception as e:
        print(f"   ❌ AI Model Error: {e}")
else:
    print(f"   ⚠️ Warning: 모델 파일 없음 ({model_path}) - AI 기능 비활성화됨")

# ==========================================
#  [공통 유틸리티]
# ==========================================
def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# ==========================================
#  [Code 1 로직] 영어/영수증 특화 함수들
# ==========================================
def fix_spelling(text):
    """영어 스펠링 교정"""
    if not text: return ""
    lines = text.split('\n')
    corrected_lines = []
    for line in lines:
        words = line.split()
        corrected_words = []
        for word in words:
            clean_word = ''.join(filter(str.isalpha, word))
            if clean_word and len(clean_word) > 2 and clean_word.lower() not in spell:
                correction = spell.correction(clean_word)
                corrected_words.append(correction if correction else word)
            else:
                corrected_words.append(word)
        corrected_lines.append(" ".join(corrected_words))
    return "\n".join(corrected_lines)

def process_english_mode(pil_image, corners=None):
    """영어 모드 전처리"""
    img_np = np.array(pil_image)
    
    # 투명도 처리
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        alpha = img_np[:, :, 3]
        rgb = img_np[:, :, :3]
        white_bg = np.full_like(rgb, 255)
        alpha_factor = alpha[:, :, np.newaxis] / 255.0
        img_np = (rgb.astype(float) * alpha_factor + white_bg.astype(float) * (1 - alpha_factor)).astype(np.uint8)

    # 투시 변환
    if corners:
        img_np = apply_perspective_transform(img_np, corners)

    if len(img_np.shape) == 3:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_np

    # 단순 확대
    img_cv = cv2.resize(img_cv, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel_sharp)

    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

# ==========================================
#  [Code 2 로직] 한글/AI 고화질 특화 함수들
# ==========================================
def group_ocr_results(results):
    """줄바꿈 그룹핑 알고리즘"""
    if not results: return ""
    boxes = []
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        cy = (tl[1] + bl[1]) / 2
        cx = (tl[0] + tr[0]) / 2
        h = bl[1] - tl[1]
        boxes.append({'text': text, 'cy': cy, 'cx': cx, 'h': h})
    
    boxes = sorted(boxes, key=lambda k: k['cy'])
    lines = []
    current_line = []
    if boxes: current_line.append(boxes[0])
    
    for i in range(1, len(boxes)):
        prev = current_line[-1]
        curr = boxes[i]
        if abs(curr['cy'] - prev['cy']) < (prev['h'] * 0.6):
            current_line.append(curr)
        else:
            current_line = sorted(current_line, key=lambda k: k['cx'])
            lines.append(current_line)
            current_line = [curr]
    if current_line:
        current_line = sorted(current_line, key=lambda k: k['cx'])
        lines.append(current_line)
        
    return "\n".join(["   ".join([item['text'] for item in line]) for line in lines])

def process_korean_mode(pil_image, corners=None, opt_grayscale=False, opt_shadow=False):
    """
    [수정됨] 한글 모드 전처리
    - 사용자 옵션(그림자/흑백)을 강제하지 않고 존중하도록 수정
    - Code 2의 우수한 전처리 파이프라인 적용
    """
    img_np = np.array(pil_image)

    # 투시 변환
    if corners:
        img_np = apply_perspective_transform(img_np, corners)

    # RGB -> BGR
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. AI 업스케일링 (가로 1000px 미만일 때만 - Code 2 기준 적용)
    h, w = img_cv.shape[:2]
    if ai_loaded and w < 1000: 
        try:
            img_cv = sr.upsample(img_cv)
        except: pass

    # 2. 그레이스케일 변환
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv

    # 3. CLAHE (대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. 노이즈 제거 (Gaussian Blur) - Code 2의 방식
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 5. 샤프닝
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # 6. 그림자 제거 (사용자 옵션이 켜져있을 때만 수행!)
    if opt_shadow:
        try:
            # 샤프닝 된 이미지 기준으로 다시 RGB 가짜 변환 후 처리 (로직상 이득)
            temp_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            rgb_planes = cv2.split(temp_rgb)
            result_planes = []
            for plane in rgb_planes:
                dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg = cv2.medianBlur(dilated, 21)
                diff = 255 - cv2.absdiff(plane, bg)
                norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(norm)
            sharpened = result_planes[0] # 그레이스케일이므로 첫 채널만 취함
        except: pass

    # 7. 흑백/컬러 반환 결정
    if opt_grayscale:
        # 흑백 모드면 전처리된 그레이스케일 그대로 반환
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
    else:
        # 컬러 모드여도 인식률을 위해 전처리된 이미지를 반환하되,
        # 원본 색감을 원하면 여기서 img_np를 반환해야 함. 
        # 하지만 OCR 인식용이므로 전처리된 이미지를 주는 게 맞음.
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

def apply_perspective_transform(img_np, corners):
    try:
        pts = np.float32([[c["x"], c["y"]] for c in corners])
        widthA = np.linalg.norm(pts[2] - pts[3])
        widthB = np.linalg.norm(pts[1] - pts[0])
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(pts[1] - pts[2])
        heightB = np.linalg.norm(pts[0] - pts[3])
        maxHeight = int(max(heightA, heightB))
        dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(img_np, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    except:
        return img_np

# ==========================================
#  [API 라우트]
# ==========================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    if "file" not in request.files: return jsonify({"error": "파일이 없습니다."}), 400
    file = request.files["file"]
    
    scan_mode = request.form.get("scan_mode", "korean")
    corners_json = request.form.get("corners")
    
    # [수정됨] 사용자 옵션 받기 (문자열 "true" -> 불리언 True)
    opt_grayscale = request.form.get("opt_grayscale", "false") == "true"
    opt_shadow = request.form.get("opt_shadow", "false") == "true"
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        corners = json.loads(corners_json) if corners_json else None
        
        processed_img = None
        ocr_text = ""
        start_time = time.time()

        if scan_mode == "english":
            print(">>> [모드 실행] 영어/스펠링체크 모드")
            # 영어 모드는 opt_shadow 등의 옵션을 내부적으로 자동 처리하거나 무시 (기존 유지)
            processed_img = process_english_mode(img, corners)
            raw_results = reader_en.readtext(np.array(processed_img), detail=1, paragraph=False)
            grouped_text = group_ocr_results(raw_results)
            ocr_text = fix_spelling(grouped_text)
            
        else: # scan_mode == "korean"
            print(">>> [모드 실행] 한글/AI고화질 모드")
            
            # [수정됨] 사용자 옵션을 함수에 전달!
            processed_img = process_korean_mode(img, corners, opt_grayscale, opt_shadow)
            
            raw_results = reader_ko.readtext(np.array(processed_img), detail=1, paragraph=False)
            ocr_text = group_ocr_results(raw_results)

        elapsed = time.time() - start_time
        print(f"처리 완료: {elapsed:.2f}초 소요")

        return jsonify({
            "text": ocr_text,
            "image_data": pil_to_base64(processed_img),
            "mode": scan_mode,
            "time": f"{elapsed:.2f}s"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/merge_pdf", methods=["POST"])
def api_merge_pdf():
    try:
        data = request.get_json(force=True)
        images = data.get("images", [])
        if not images: return jsonify({"error": "이미지가 없습니다."}), 400
        pil_images = []
        for url in images:
            b64 = url.split(",", 1)[1] if "," in url else url
            pil_images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
        buf = io.BytesIO()
        pil_images[0].save(buf, format="PDF", save_all=True, append_images=pil_images[1:])
        buf.seek(0)
        return jsonify({"pdf_data": "data:application/pdf;base64," + base64.b64encode(buf.read()).decode("utf-8")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
