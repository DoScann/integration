import io
import os
import base64
import json
import time

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
from paddleocr import PaddleOCR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

print("Loading PaddleOCR...")
ocr = PaddleOCR(
    lang="korean",
    use_angle_cls=True,        # 글자 기울어져 있을 때 보정
    det_db_box_thresh=0.3,     # 너무 약한 글자 박스도 조금 더 잡게
    det_db_unclip_ratio=1.6    # 박스 살짝 넓게
)


@app.route("/")
def index():
    return render_template("index1.html")


def resize_for_ocr(img_rgb):
    """OCR 인식용으로만 사용하는 리사이즈 함수"""
    h, w = img_rgb.shape[:2]

    max_side = 2000
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_rgb = cv2.resize(
            img_rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        h, w = img_rgb.shape[:2]

    if max(h, w) < 900:
        img_rgb = cv2.resize(
            img_rgb,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

    return img_rgb


def remove_shadow_rgb(img_rgb):
    """컬러 유지하면서 그림자 제거"""
    try:
        rgb_planes = cv2.split(img_rgb)
        result_planes = []

        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(
                diff,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            )
            result_planes.append(norm)

        result = cv2.merge(result_planes)
        return result
    except Exception:
        return img_rgb


def apply_binary_if_needed(img_rgb, use_binary):
    """이진화가 필요할 때만 적용 (현재는 인식용에서 내부적으로 사용 가능)"""
    if not use_binary:
        return img_rgb

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)


def warp_perspective_if_needed(img_rgb, corners):
    """사용자가 선택한 네 점 기준으로 문서 투시 보정"""
    if not corners or len(corners) != 4:
        return img_rgb

    try:
        pts = np.float32([[c["x"], c["y"]] for c in corners])

        width_a = np.linalg.norm(pts[2] - pts[3])
        width_b = np.linalg.norm(pts[1] - pts[0])
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(pts[1] - pts[2])
        height_b = np.linalg.norm(pts[0] - pts[3])
        max_height = int(max(height_a, height_b))

        if max_width <= 0 or max_height <= 0:
            return img_rgb

        dst = np.float32(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ]
        )

        m = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            img_rgb,
            m,
            (max_width, max_height),
            flags=cv2.INTER_CUBIC,
        )
        return warped
    except Exception:
        return img_rgb


def make_display_image(pil_image, corners=None, opt_grayscale=False, opt_shadow=False):
    """
    최종적으로 사용자에게 보여줄 이미지 생성
    사용자가 체크한 필터만 적용
    """
    img_rgb = np.array(pil_image)

    # 문서 영역 보정
    img_rgb = warp_perspective_if_needed(img_rgb, corners)

    # 그림자 제거 필터
    if opt_shadow:
        img_rgb = remove_shadow_rgb(img_rgb)

    # 흑백 필터
    if opt_grayscale:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(img_rgb)


def enhance_image_for_ocr(img_rgb):
    """
    OCR 인식 전용 전처리 (영수증 특화 버전)
      1. 업스케일 x2 (CUBIC)
      2. 그레이 변환
      3. CLAHE로 글자 대비만 살짝 올리기
      4. 약한 블러 후 아주 약한 샤프닝
    이진화는 하지 않음.
    """
    # 1. 업스케일 x2
    img = cv2.resize(
        img_rgb,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # 2. 그레이 변환
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. CLAHE로 글자 대비만 살짝
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)

    # 4. 아주 약한 블러로 노이즈만 살짝 줄이고
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. 약한 샤프닝
    sharp_kernel = np.array(
        [
            [0, -0.5, 0],
            [-0.5, 3.0, -0.5],
            [0, -0.5, 0],
        ],
        dtype=np.float32,
    )
    gray = cv2.filter2D(gray, -1, sharp_kernel)

    # PaddleOCR는 RGB도 받으니까 다시 3채널로
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)



def make_ocr_image(pil_image, corners=None):
    """
    OCR 인식 전용 이미지
    - 문서 투시 보정
    - 업스케일 + 라이트 전처리
    """
    img_rgb = np.array(pil_image)

    # 1) 문서 영역 보정
    img_rgb = warp_perspective_if_needed(img_rgb, corners)

    # 2) 기본 사이즈 조정
    img_rgb = resize_for_ocr(img_rgb)

    # 3) 업스케일 + 밝기보정 + 샤프닝
    img_rgb = enhance_image_for_ocr(img_rgb)

    return Image.fromarray(img_rgb)



def group_ocr_results(results):
    """(bbox, text, score) 리스트를 이미지 좌표 기준으로 줄 정렬해 텍스트로 변환"""
    if not results:
        return ""

    boxes = []

    for bbox, text, prob in results:
        tl = bbox[0]
        tr = bbox[1]
        br = bbox[2]
        bl = bbox[3]

        center_y = (tl[1] + bl[1]) / 2.0
        center_x = (tl[0] + tr[0]) / 2.0
        height = abs(bl[1] - tl[1]) + 1e-6

        boxes.append(
            {
                "text": text,
                "cy": center_y,
                "cx": center_x,
                "h": height,
            }
        )

    boxes.sort(key=lambda k: k["cy"])

    lines = []
    current_line = []

    if boxes:
        current_line.append(boxes[0])

    for i in range(1, len(boxes)):
        prev = current_line[-1]
        curr = boxes[i]

        same_line = abs(curr["cy"] - prev["cy"]) < (prev["h"] * 0.6)

        if same_line:
            current_line.append(curr)
        else:
            current_line.sort(key=lambda k: k["cx"])
            lines.append(current_line)
            current_line = [curr]

    if current_line:
        current_line.sort(key=lambda k: k["cx"])
        lines.append(current_line)

    output_text = ""

    for line in lines:
        line_str = "   ".join(item["text"] for item in line)
        output_text += line_str + "\n"

    return output_text


def run_ocr_on_image_fast(pil_image):
    """
    전처리된 PIL 이미지를 받아 PaddleOCR 실행
    가능한 경우 bbox 정보를 사용해 이미지 상 위치 기준으로 줄을 정렬해서 반환하고,
    bbox를 못 찾으면 문자열만 모아서 반환
    맨 앞에 나오는 min general 같은 노이즈 라인은 자동으로 필터링
    """
    img_rgb = np.array(pil_image)

    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_rgb

    raw = ocr.ocr(img_bgr)
    print("PaddleOCR raw type:", type(raw))

    if raw is None:
        print("raw is None")
        return ""

    bbox_results = []   # (bbox, text, score)
    loose_texts = []    # bbox 없는 텍스트 (fallback)

    def is_number(x):
        import numpy as _np
        return isinstance(x, (int, float, _np.integer, _np.floating))

    def is_point(p):
        if not isinstance(p, (list, tuple, np.ndarray)):
            return False
        if len(p) != 2:
            return False
        return is_number(p[0]) and is_number(p[1])

    def is_bbox(b):
        if not isinstance(b, (list, tuple, np.ndarray)):
            return False
        if len(b) != 4:
            return False
        return all(is_point(pt) for pt in b)

    def add_result(bbox, text, score):
        if not is_bbox(bbox):
            return
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            return
        try:
            s = float(score)
        except Exception:
            s = 0.0
        bbox_results.append((bbox, text, s))

    def walk(obj):
        """
        raw 전체를 재귀적으로 돌면서
        dict / list / tuple 안에서 (bbox, text, score)를 최대한 많이 찾고,
        못 찾는 문자열은 loose_texts에 넣는다
        """
        # dict
        if isinstance(obj, dict):
            bbox = obj.get("points") or obj.get("bbox") or obj.get("box")
            text = obj.get("transcription") or obj.get("text")
            score = obj.get("score") or obj.get("confidence")

            if bbox is not None and text is not None:
                add_result(bbox, text, score if score is not None else 0.0)
                return

            for v in obj.values():
                walk(v)
            return

        # list / tuple
        if isinstance(obj, (list, tuple)):
            # [bbox, (text, score)] 포맷
            if (
                len(obj) == 2
                and is_bbox(obj[0])
                and isinstance(obj[1], (list, tuple))
                and len(obj[1]) >= 1
            ):
                t = obj[1][0]
                sc = obj[1][1] if len(obj[1]) >= 2 else 0.0
                add_result(obj[0], t, sc)
                return

            # [bbox, text, score, ...] 포맷
            if (
                len(obj) >= 2
                and is_bbox(obj[0])
                and isinstance(obj[1], str)
            ):
                t = obj[1]
                sc = obj[2] if len(obj) >= 3 else 0.0
                add_result(obj[0], t, sc)
                return

            for v in obj:
                walk(v)
            return

        # 문자열
        if isinstance(obj, str):
            if obj.strip():
                loose_texts.append(obj)
            return

        # 나머지 타입은 무시
        return

    # 전체 결과 스캔
    walk(raw)

    print(f"bbox_results: {len(bbox_results)}, loose_texts: {len(loose_texts)}")

    # 우선순위 1  bbox 있는 결과들
    if bbox_results:
        result = group_ocr_results(bbox_results)
    # 우선순위 2  bbox 없고 문자열만 있는 경우
    elif loose_texts:
        result = "\n".join(loose_texts)
    else:
        try:
            if isinstance(raw, list) and raw:
                print("raw[0] keys:", list(raw[0].keys()))
            else:
                print("raw sample:", repr(raw)[:500])
        except Exception:
            pass
        return ""

    # 맨 앞 노이즈 라인 필터링
    def is_noise_line(text):
        t = text.strip().lower()
        if t in ["min", "general", "min general", "doc", "page", "text", "line"]:
            return True
        if len(t) <= 2:
            return True
        return False

    lines = result.splitlines()
    cleaned = []
    for idx, line in enumerate(lines):
        if idx == 0:   # 첫 줄 삭제
            continue
        if is_noise_line(line):
            continue
        cleaned.append(line)


    result = "\n".join(cleaned).strip()

    return result


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]

    corners_json = request.form.get("corners", None)
    opt_grayscale = request.form.get("opt_grayscale", "false") == "true"
    opt_shadow = request.form.get("opt_shadow", "false") == "true"

    try:
        start_time = time.time()

        img_orig = Image.open(io.BytesIO(file.read())).convert("RGB")
        corners = json.loads(corners_json) if corners_json else None

        # 사용자에게 보여줄 이미지
        display_img = make_display_image(
            img_orig,
            corners=corners,
            opt_grayscale=opt_grayscale,
            opt_shadow=opt_shadow,
        )

        # OCR 인식용 이미지
        ocr_img = make_ocr_image(
            img_orig,
            corners=corners,
        )

        print(f"Processing time (preprocess): {time.time() - start_time:.2f}s")

        ocr_text = run_ocr_on_image_fast(ocr_img)

        # 결과 표시용 이미지를 PNG로 인코딩
        buf = io.BytesIO()
        display_img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify(
            {
                "text": ocr_text,
                "image_data": "data:image/png;base64," + b64,
            }
        )
    except Exception as e:
        print(f"Error in /api/ocr: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/merge_pdf", methods=["POST"])
def api_merge_pdf():
    try:
        data = request.get_json(force=True)
        pil_images = []

        for d in data.get("images", []):
            b64 = d.split(",", 1)[1] if "," in d else d
            pil_images.append(
                Image.open(
                    io.BytesIO(base64.b64decode(b64))
                ).convert("RGB")
            )

        if not pil_images:
            return jsonify({"error": "no images"}), 400

        buf = io.BytesIO()
        pil_images[0].save(
            buf,
            format="PDF",
            save_all=True,
            append_images=pil_images[1:],
        )
        buf.seek(0)
        pdf_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify(
            {
                "pdf_data": "data:application/pdf;base64," + pdf_b64,
            }
        )
    except Exception as e:
        print(f"Error in /api/merge_pdf: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
