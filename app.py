from flask import Flask, render_template, Response, jsonify, request
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

# =========================
# CONFIG
# =========================
DEFAULT_COUNTRY_CODE = "+880"   # Bangladesh (change if needed)

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
# PREPROCESSING
# =========================
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)

# =========================
# OCR FUNCTIONS
# =========================
def easyocr_with_boxes(image):
    """
    Returns:
        texts: list[str]
        boxes: list[list[tuple]]
    """
    results = reader.readtext(image)
    texts, boxes = [], []

    for bbox, text, conf in results:
        if text.strip():
            texts.append(text.strip())
            boxes.append(bbox)

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
# PHONE EXTRACTION (FIXED)
# =========================
def extract_phone_numbers(texts, default_cc):
    joined = " ".join(texts)

    candidates = re.findall(
        r"(?:\+?\d{1,3})?[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}",
        joined
    )

    phones = set()

    for c in candidates:
        digits = re.sub(r"\D", "", c)

        # Bangladesh local format: 01XXXXXXXXX
        if len(digits) == 11 and digits.startswith("0"):
            phones.add(default_cc + digits[1:])

        # With country code already
        elif len(digits) >= 12:
            phones.add("+" + digits)

        # Missing country code
        elif 9 <= len(digits) <= 10:
            phones.add(default_cc + digits)

    return list(phones)

# =========================
# ADDRESS EXTRACTION (FIXED)
# =========================
def extract_address(texts):
    address_keywords = [
        "road", "rd", "street", "st", "sector", "block", "area",
        "avenue", "ave", "lane", "ln", "floor", "building",
        "dhaka", "bangladesh", "city", "zip"
    ]

    address_lines = []

    for t in texts:
        lower = t.lower()
        if any(k in lower for k in address_keywords):
            address_lines.append(t)

    if len(address_lines) >= 2:
        return " ".join(address_lines)

    if address_lines:
        return address_lines[0]

    return None

# =========================
# CARD TYPE DETECTION
# =========================
def detect_card_type(texts):
    joined = " ".join(texts).lower()
    visiting, idc = 0, 0

    if extract_phone_numbers(texts, DEFAULT_COUNTRY_CODE):
        visiting += 2
    if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", joined):
        visiting += 3
    if "www" in joined or ".com" in joined:
        visiting += 2

    if re.search(r"\d{2}[-/]\d{2}[-/]\d{4}", joined):
        idc += 3
    if re.search(r"\b\d{10,17}\b", joined):
        idc += 3

    if idc > visiting:
        return "id_card"
    if visiting > idc:
        return "visiting_card"
    return "unknown"

# =========================
# FIELD EXTRACTION
# =========================
def extract_fields(texts, card_type):
    joined = "\n".join(texts)
    data = {}

    phones = extract_phone_numbers(texts, DEFAULT_COUNTRY_CODE)
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", joined)
    websites = re.findall(r"(www\.[^\s]+)", joined)
    dob = re.findall(r"\d{2}[-/]\d{2}[-/]\d{4}", joined)
    ids = re.findall(r"\b\d{10,17}\b", joined)
    address = extract_address(texts)

    if phones:
        data["phones"] = phones
    if emails:
        data["emails"] = emails
    if websites:
        data["websites"] = websites
    if dob:
        data["date_of_birth"] = dob[0]
    if ids:
        data["id_number"] = ids[0]
    if address:
        data["address"] = address

    names = [t for t in texts if t.isupper() and 2 <= len(t.split()) <= 4]
    if names:
        data["name"] = names[0]

    if card_type == "visiting_card":
        for t in texts:
            if any(x in t.lower() for x in ["manager", "engineer", "director", "officer"]):
                data["designation"] = t
                break

    return data

# =========================
# LIVE STREAM WITH BOXES
# =========================
def generate_frames():
    global latest_result
    count = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        if count % 10 == 0:
            easy_texts, easy_boxes = easyocr_with_boxes(frame)

            texts = merge_texts(
                easy_texts,
                tesseract_text(frame),
                tesseract_text(preprocess(frame))
            )

            detected = bool(texts)

            if detected:
                cv2.putText(frame, "TEXT DETECTED",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                # Draw EasyOCR bounding boxes
                for box in easy_boxes:
                    pts = np.array(box, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            card_type = detect_card_type(texts)
            extracted = extract_fields(texts, card_type)

            with lock:
                latest_result = {
                    "detected": detected,
                    "card_type": card_type,
                    "raw_texts": texts,
                    "extracted_data": extracted
                }

        count += 1

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

@app.route("/api/ocr", methods=["GET"])
def get_ocr():
    with lock:
        return jsonify(latest_result)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
