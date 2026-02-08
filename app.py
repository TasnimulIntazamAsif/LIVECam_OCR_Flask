from flask import Flask, render_template, Response, jsonify
import cv2
import easyocr
import pytesseract
import numpy as np
import threading
import atexit
import re

# =========================
# APP
# =========================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

DEFAULT_COUNTRY_CODE = "+880"

# =========================
# CAMERA
# =========================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Cannot open webcam")

def release_camera():
    if camera.isOpened():
        camera.release()

atexit.register(release_camera)

# =========================
# OCR
# =========================
reader = easyocr.Reader(['en'], gpu=False)

# =========================
# GLOBAL
# =========================
latest_result = {}
lock = threading.Lock()

# =========================
# FAST CARD PRESENCE CHECK (FIXED)
# =========================
def card_present_in_roi(roi):
    """
    Fast & reliable:
    If ROI has enough edges → card is present
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    edge_pixels = cv2.countNonZero(edges)
    roi_pixels = roi.shape[0] * roi.shape[1]

    edge_ratio = edge_pixels / roi_pixels

    return edge_ratio > 0.015  # tuned threshold

# =========================
# OCR HELPERS
# =========================
def easyocr_with_boxes(image):
    results = reader.readtext(image)
    texts, boxes = [], []
    for box, text, conf in results:
        if text.strip():
            texts.append(text.strip())
            boxes.append(box)
    return texts, boxes

def tesseract_text(image):
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6"
    )
    return [t.strip() for t in data["text"] if t.strip()]

def merge_texts(*lists):
    seen = []
    for lst in lists:
        for t in lst:
            if t not in seen:
                seen.append(t)
    return seen

# =========================
# EXTRACTION
# =========================
def extract_phone_numbers(texts):
    joined = " ".join(texts)
    matches = re.findall(r"\+?\d[\d\s\-]{8,15}", joined)
    phones = set()

    for m in matches:
        digits = re.sub(r"\D", "", m)
        if digits.startswith("0") and len(digits) == 11:
            phones.add(DEFAULT_COUNTRY_CODE + digits[1:])
        elif digits.startswith("880"):
            phones.add("+" + digits)

    return list(phones)

def extract_address(texts):
    keywords = ["road", "street", "sector", "block", "dhaka", "bangladesh"]
    lines = [t for t in texts if any(k in t.lower() for k in keywords)]
    return " ".join(lines) if lines else None

# =========================
# LIVE STREAM (FIXED)
# =========================
def generate_frames():
    global latest_result

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape

        # -------- FIXED PLACEMENT BOX --------
        box_w, box_h = int(w * 0.75), int(h * 0.45)
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "PLACE CARD INSIDE BOX",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

        roi = frame[y1:y2, x1:x2]

        detected = False
        texts = []
        extracted = {}

        if card_present_in_roi(roi):
            cv2.putText(frame, "CARD DETECTED - SCANNING",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            easy_texts, easy_boxes = easyocr_with_boxes(roi)
            tess_texts = tesseract_text(roi)
            texts = merge_texts(easy_texts, tess_texts)
            detected = bool(texts)

            for box in easy_boxes:
                pts = np.array(box, dtype=np.int32)
                pts[:, 0] += x1
                pts[:, 1] += y1
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            extracted = {
                "phones": extract_phone_numbers(texts),
                "address": extract_address(texts)
            }

        else:
            cv2.putText(frame, "NO CARD IN BOX",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 3)

        with lock:
            latest_result = {
                "detected": detected,
                "raw_texts": texts,
                "extracted_data": extracted
            }

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/ocr")
def get_ocr():
    with lock:
        return jsonify(latest_result)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
