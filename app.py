import io
import os
import base64
import json

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import easyocr
import cv2

# app.py가 있는 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask 앱 (templates 폴더 명시)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# EasyOCR 리더 (한국어 + 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=False)


@app.route("/")
def index():
    return render_template("index.html")


def run_ocr_on_image(pil_image: Image.Image) -> str:
    """PIL.Image -> EasyOCR 텍스트"""
    img_np = np.array(pil_image)
    result = reader.readtext(img_np, detail=0)  # 텍스트만
    text = "\n".join(result)
    return text


def process_image_core(pil_image,
                       corners=None,
                       opt_grayscale=False,
                       opt_shadow=False):
    """
    전처리 핵심 함수
    - corners: 4점 [{x, y}, ...] 있으면 투시 변환(기울기 보정)
    - opt_grayscale: 흑백 변환
    - opt_shadow: 간단한 그림자 제거
    반환: 전처리된 PIL.Image(RGB)
    """
    img_np = np.array(pil_image)  # RGB

    # 1) 4점 투시 변환 (기울기 보정)
    if corners and len(corners) == 4:
        try:
            pts = np.float32([[c["x"], c["y"]] for c in corners])

            widthA = np.linalg.norm(pts[2] - pts[3])  # BR-BL
            widthB = np.linalg.norm(pts[1] - pts[0])  # TR-TL
            maxWidth = int(max(widthA, widthB))

            heightA = np.linalg.norm(pts[1] - pts[2])  # TR-BR
            heightB = np.linalg.norm(pts[0] - pts[3])  # TL-BL
            maxHeight = int(max(heightA, heightB))

            if maxWidth > 0 and maxHeight > 0:
                dst = np.float32([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ])

                M = cv2.getPerspectiveTransform(pts, dst)
                img_np = cv2.warpPerspective(
                    img_np, M,
                    (maxWidth, maxHeight),
                    flags=cv2.INTER_CUBIC
                )
        except Exception:
            # 실패하면 그냥 원본 유지
            pass

    # 2) 그림자 제거 (옵션, 간단한 방식)
    if opt_shadow:
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(gray, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX)
            img_np = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass

    # 3) 흑백 변환 (옵션)
    if opt_grayscale:
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        except Exception:
            pass

    processed_pil = Image.fromarray(img_np)
    return processed_pil


def pil_to_data_url(pil_image):
    """PIL.Image -> data:image/png;base64,... 문자열"""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """
    프론트에서 파일 + 옵션 + 코너 정보 받아서
    1) 전처리(흑백/그림자/투시보정)
    2) EasyOCR
    3) 전처리된 PNG(data URL) + 텍스트 반환
    """
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    file_bytes = file.read()

    corners_json = request.form.get("corners", None)
    opt_grayscale = request.form.get("opt_grayscale", "false") == "true"
    opt_shadow = request.form.get("opt_shadow", "false") == "true"

    mime = file.mimetype or ""
    ext = os.path.splitext(file.filename)[1].lower()

    # PDF는 아직 이미지 변환 없이 안내만
    if mime == "application/pdf" or ext == ".pdf":
        return jsonify({
            "text": "현재 버전은 PDF OCR을 직접 지원하지 않습니다.\nPDF를 이미지(JPG/PNG)로 변환해서 업로드해 주세요.",
            "image_data": None
        })

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        corners = None
        if corners_json:
            try:
                corners = json.loads(corners_json)
            except Exception:
                corners = None

        processed = process_image_core(
            img,
            corners=corners,
            opt_grayscale=opt_grayscale,
            opt_shadow=opt_shadow
        )

        ocr_text = run_ocr_on_image(processed)
        processed_image_data_url = pil_to_data_url(processed)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "text": ocr_text,
        "image_data": processed_image_data_url
    })


@app.route("/api/merge_pdf", methods=["POST"])
def api_merge_pdf():
    """
    data URL PNG들을 하나의 PDF로 병합해서
    data:application/pdf;base64,... 형태로 반환
    """
    try:
        data = request.get_json(force=True)
        images_data = data.get("images", [])
        if not images_data:
            return jsonify({"error": "no images"}), 400

        pil_images = []
        for data_url in images_data:
            if "," in data_url:
                _, b64 = data_url.split(",", 1)
            else:
                b64 = data_url
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_images.append(img)

        if not pil_images:
            return jsonify({"error": "no valid images"}), 400

        first = pil_images[0]
        rest = pil_images[1:]

        buf = io.BytesIO()
        first.save(buf, format="PDF", save_all=True, append_images=rest)
        buf.seek(0)
        pdf_b64 = base64.b64encode(buf.read()).decode("utf-8")
        data_url = "data:application/pdf;base64," + pdf_b64

        return jsonify({"pdf_data": data_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)