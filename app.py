from flask import Flask, render_template, Response, jsonify, request
import cv2
import easyocr
import pytesseract
import numpy as np
import threading
import atexit

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# =========================
# CAMERA SETUP
# =========================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

def release_camera():
    if camera.isOpened():
        camera.release()

atexit.register(release_camera)

# =========================
# OCR SETUP
# =========================
reader = easyocr.Reader(['en'], gpu=False)

# Uncomment if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================
# GLOBAL STORAGE
# =========================
latest_texts = {}
latest_boxes = {}
lock = threading.Lock()

# =========================
# SAFE JSON CONVERTER
# =========================
def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return [to_python_types(i) for i in obj]
    elif isinstance(obj, np.generic):  # np.int32, np.float32, etc.
        return obj.item()
    else:
        return obj

# =========================
# TESSERACT OCR
# =========================
def run_tesseract_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6"
    )

    texts = []
    boxes = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if text and conf > 50:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            texts.append(text)
            boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

    return texts, boxes

# =========================
# LIVE CAMERA STREAM
# =========================
def generate_frames():
    global latest_texts, latest_boxes
    frame_count = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 10 == 0:
            # -------- EasyOCR --------
            easy_results = reader.readtext(gray)
            easy_texts = []
            easy_boxes = []

            for (bbox, text, prob) in easy_results:
                if prob > 0.5:
                    easy_texts.append(text)
                    easy_boxes.append(
                        [[int(x), int(y)] for x, y in bbox]
                    )

                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        text,
                        (int(pts[0][0]), int(pts[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # -------- Tesseract --------
            tess_texts, tess_boxes = run_tesseract_ocr(frame)

            # -------- Detection Signal --------
            detected = bool(easy_texts or tess_texts)

            if detected:
                cv2.putText(
                    frame,
                    "TEXT DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            with lock:
                latest_texts = {
                    "detected": detected,
                    "easyocr": easy_texts,
                    "tesseract": tess_texts
                }
                latest_boxes = {
                    "easyocr": easy_boxes,
                    "tesseract": tess_boxes
                }

        frame_count += 1

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes() +
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
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# -------- GET OCR --------
@app.route("/api/ocr", methods=["GET"])
def get_ocr():
    with lock:
        return jsonify(
            to_python_types({
                "status": "success",
                "detected": latest_texts.get("detected", False),
                "easyocr_texts": latest_texts.get("easyocr", []),
                "tesseract_texts": latest_texts.get("tesseract", []),
                "boxes": latest_boxes
            })
        )

# -------- POST OCR --------
@app.route("/api/ocr", methods=["POST"])
def post_ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    np_img = np.frombuffer(request.files["image"].read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    easy_results = reader.readtext(image)
    easy_output = []

    for (bbox, text, prob) in easy_results:
        if prob > 0.5:
            easy_output.append({
                "text": text,
                "confidence": float(prob),
                "bounding_box": [[int(x), int(y)] for x, y in bbox]
            })

    tess_texts, _ = run_tesseract_ocr(image)
    detected = bool(easy_output or tess_texts)

    return jsonify(
        to_python_types({
            "status": "success",
            "detected": detected,
            "easyocr": easy_output,
            "tesseract": tess_texts
        })
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
