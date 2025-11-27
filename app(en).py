import io
import os
import base64
import json
import logging
import traceback

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
import easyocr

# [추가] 스펠링 체크 라이브러리
from spellchecker import SpellChecker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# --- EasyOCR 초기화 ---
print("EasyOCR 모델 로딩 중... (잠시만 기다려주세요)")
# [변경] 'ko'를 제거하고 'en'만 남겨 영어 인식률을 극대화합니다.
reader = easyocr.Reader(['en'], gpu=False)

# [추가] 영어 스펠링 체커 초기화
spell = SpellChecker()
print("EasyOCR & SpellChecker 로드 완료!")


@app.route("/")
def index():
    return render_template("index.html")


def fix_spelling(text):
    """
    [후처리] OCR 결과 텍스트의 오타를 교정하는 함수
    """
    if not text:
        return ""

    lines = text.split('\n')
    corrected_lines = []

    for line in lines:
        words = line.split()
        corrected_words = []
        
        for word in words:
            # 특수문자가 섞인 경우 등을 대비해 순수 알파벳인 경우만 필터링
            clean_word = ''.join(filter(str.isalpha, word))
            
            # 3글자 이상인 단어만 검사 (짧은 단어는 오교정 확률 높음)
            if clean_word and len(clean_word) > 2:
                # 사전에 없는 단어인지 확인
                if clean_word.lower() not in spell:
                    # 가장 유력한 정답 단어 추천
                    correction = spell.correction(clean_word)
                    
                    # 교정된 단어가 있으면 교체
                    if correction:
                        corrected_words.append(correction)
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word) # 사전에 있으면 그대로 둠
            else:
                corrected_words.append(word) # 알파벳 아니면 그대로 둠
        
        corrected_lines.append(" ".join(corrected_words))

    return "\n".join(corrected_lines)


def run_ocr_on_image(pil_image: Image.Image) -> str:
    """EasyOCR 실행 함수"""
    img_np = np.array(pil_image)
    
    try:
        result = reader.readtext(
            img_np, 
            detail=0,
            paragraph=True,     # 문단 인식 모드 (자연스러운 연결)
            canvas_size=4096,   # 고해상도 처리
            mag_ratio=1.0,      # 이미 전처리에서 확대함
            adjust_contrast=0.5,
            x_ths=1.0,          # 가로 간격 허용치 (띄어쓰기 관대하게)
            y_ths=0.5           # 세로 간격 허용치 (줄바꿈 관대하게)
        )
        
        raw_text = "\n\n".join(result)
        
        # [추가] 스펠링 교정 수행
        final_text = fix_spelling(raw_text)
        
        return final_text
        
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return "텍스트를 인식할 수 없습니다."


def handle_transparency(img_cv):
    """[추가] 투명 배경(Alpha) 처리 -> 흰색 배경으로 병합"""
    if img_cv.shape[2] == 4:  # 채널이 4개면(RGBA) 투명도가 있다는 뜻
        # 알파 채널 분리
        alpha_channel = img_cv[:, :, 3]
        rgb_channels = img_cv[:, :, :3]

        # 마스크 생성 (투명한 부분 = 0)
        # 투명하지 않은 부분은 그대로(1.0), 투명한 부분은 흰색(255)으로 대체 계산
        
        # 간단하게: 투명한 픽셀(alpha < 255)을 흰색으로 덮어씌우기
        white_bg_img = np.full_like(rgb_channels, 255)
        
        # 알파값 정규화 (0.0 ~ 1.0)
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        
        # 합성이미지 = (원본RGB * 알파) + (흰배경 * (1-알파))
        base = rgb_channels.astype(float) * alpha_factor
        white = white_bg_img.astype(float) * (1 - alpha_factor)
        final_img = (base + white).astype(np.uint8)
        
        return final_img
    return img_cv


def process_image_core(pil_image, corners=None, opt_grayscale=False, opt_shadow=False):
    """
    [극한의 전처리 파이프라인]
    1. 투명도 제거 (흰색 배경)
    2. 투시 변환
    3. 2배~3배 확대
    4. 양방향 필터 (노이즈 제거 + 엣지 보존)
    5. 모폴로지 닫기 (점선 잇기 - 영수증 핵심)
    6. CLAHE (스마트 대비)
    7. 선명화
    """
    img_np = np.array(pil_image)
    
    # 1. [신규] 투명도 처리 (PNG 대응)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_np = handle_transparency(img_np)
    
    # RGB -> BGR (OpenCV용)
    if len(img_np.shape) == 3:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_np

    # 2. 투시 변환
    if corners and len(corners) == 4:
        try:
            pts = np.float32([[c["x"], c["y"]] for c in corners])
            widthA = np.linalg.norm(pts[2] - pts[3])
            widthB = np.linalg.norm(pts[1] - pts[0])
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(pts[1] - pts[2])
            heightB = np.linalg.norm(pts[0] - pts[3])
            maxHeight = int(max(heightA, heightB))

            if maxWidth > 0 and maxHeight > 0:
                dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])
                M = cv2.getPerspectiveTransform(pts, dst)
                img_cv = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
        except: pass

    # 3. 확대 (Upscaling)
    scale = 2.0
    img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 4. [신규] 양방향 필터 (Bilateral Filter)
    # 종이 질감(노이즈)은 없애고 글자 획(엣지)은 살립니다.
    # d=9 (직경), sigmaColor=75 (색공간), sigmaSpace=75 (좌표공간)
    try:
        img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
    except: pass

    # 그레이스케일 변환
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv

    # 5. [신규] 모폴로지 닫기 (Morphological Closing)
    # 도트 폰트(.....) 사이의 구멍을 잉크로 메워줍니다. 영수증 인식률에 큰 도움 됩니다.
    try:
        # 커널 크기 (2, 2): 너무 크면 글자가 뭉개지므로 아주 작게 설정
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    except: pass

    # 6. CLAHE (스마트 대비) - 그림자 제거 옵션과 상관없이 약하게라도 적용하면 좋음
    if opt_shadow:
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except: pass
    else:
        # 옵션이 꺼져있어도 기본적으로 약하게 적용 (전반적 가독성 향상)
        try:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except: pass

    # 7. 선명화 (Sharpening)
    try:
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel_sharp)
    except: pass

    # 8. 최종 변환 (흑백 옵션)
    if opt_grayscale:
        img_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) # 보여줄 땐 RGB 형식의 흑백
    else:
        # 컬러로 복구하려면 원본 색감이 필요하지만, 위에서 이미 변형이 많이 일어남.
        # 인식용으로는 흑백 처리된 결과가 더 좋으므로, 여기서는 처리된 흑백 이미지를 반환합니다.
        img_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(img_result)


def pil_to_data_url(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    if "file" not in request.files: return jsonify({"error": "파일이 없습니다."}), 400
    file = request.files["file"]
    if file.filename == "": return jsonify({"error": "파일명이 비어있습니다."}), 400
    
    file_bytes = file.read()
    corners_json = request.form.get("corners", None)
    opt_grayscale = request.form.get("opt_grayscale", "false") == "true"
    opt_shadow = request.form.get("opt_shadow", "false") == "true"

    if file.mimetype == "application/pdf" or file.filename.lower().endswith(".pdf"):
        return jsonify({"text": "PDF는 지원하지 않습니다.", "image_data": None})

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        corners = None
        if corners_json:
            try: corners = json.loads(corners_json)
            except: pass

        processed = process_image_core(img, corners, opt_grayscale, opt_shadow)
        ocr_text = run_ocr_on_image(processed)
        processed_data_url = pil_to_data_url(processed)

        return jsonify({
            "text": ocr_text,
            "image_data": processed_data_url
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
            if "," in url: _, b64 = url.split(",", 1)
            else: b64 = url
            img_bytes = base64.b64decode(b64)
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        if not pil_images: return jsonify({"error": "유효한 이미지가 없습니다."}), 400

        buf = io.BytesIO()
        pil_images[0].save(buf, format="PDF", save_all=True, append_images=pil_images[1:])
        buf.seek(0)
        return jsonify({"pdf_data": "data:application/pdf;base64," + base64.b64encode(buf.read()).decode("utf-8")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
