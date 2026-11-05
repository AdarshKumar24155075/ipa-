"""Flask backend for the Image Enhancement and Restoration System.
Run:  pip install -r requirements.txt && python app.py   ->  http://127.0.0.1:5000
"""
import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
try:
    from flask_cors import CORS
except ImportError:  # CORS is optional (only needed if the UI is hosted on another origin)
    CORS = lambda app: app

import enhancer as E

app = Flask(__name__, static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads


def read_image(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image file")
    h, w = img.shape[:2]
    if max(h, w) > 2000:  # keep CPU time bounded
        s = 2000 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def f(name, default, cast=float):
    try:
        return cast(request.form.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify(status="ok", opencv=cv2.__version__)


@app.route("/api/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify(error="No image uploaded (field name: image)"), 400
    try:
        img = read_image(request.files["image"])
        mask = None
        if "mask" in request.files:  # optional user-drawn mask, white = damaged
            m = read_image(request.files["mask"])
            mask = cv2.threshold(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = E.enhance_pipeline(
            img,
            denoise_strength=f("denoise", 8),
            deblur_sigma=f("deblur", 0),
            contrast_clip=f("contrast", 1.5),
            white_balance=request.form.get("white_balance") == "true",
            gamma_val=f("gamma", 1.0),
            sharpen=f("sharpen", 0.5),
            scale=f("scale", 1, int),
            mask=mask,
            auto_scratch=request.form.get("auto_scratch") == "true",
        )
        ok, buf = cv2.imencode(".png", out)
        return jsonify(
            image="data:image/png;base64," + base64.b64encode(buf).decode(),
            width=out.shape[1], height=out.shape[0],
        )
    except Exception as e:  # return readable error to the UI
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
