"""
AI Learning Support Platform for Children with Dyscalculia & Dysgraphia
Flask Backend - app.py
Author: Senior AI Engineer
"""

import os
import io
import re
import base64
import logging
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ─── Lazy-load heavy models ──────────────────────────────────────────────────
_digit_model = None
_ocr_available = False
_speech_model = None
_speech_processor = None

def get_digit_model():
    global _digit_model
    if _digit_model is None:
        from model import DigitClassifier
        _digit_model = DigitClassifier()
        logger.info("Digit classifier loaded.")
    return _digit_model

def get_speech_pipeline():
    global _speech_model, _speech_processor
    if _speech_model is None:
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch
            _speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            _speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            _speech_model.eval()
            logger.info("Wav2Vec2 loaded.")
        except Exception as e:
            logger.warning(f"Speech model unavailable: {e}")
    return _speech_processor, _speech_model


# ─── Utilities ───────────────────────────────────────────────────────────────

def decode_image_b64(data_url: str) -> np.ndarray:
    """Decode base64 data URL to OpenCV image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def preprocess_canvas(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Grayscale
    2. Gaussian blur (denoise)
    3. Adaptive threshold (handles varying lighting)
    4. Morphological cleanup
    5. Center digit with padding
    6. Resize to 28x28
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find bounding box of digit content
    coords = cv2.findNonZero(cleaned)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = cleaned[y:y+h, x:x+w]
        # Add 20% padding
        pad = int(max(w, h) * 0.2)
        digit_padded = cv2.copyMakeBorder(digit_crop, pad, pad, pad, pad,
                                          cv2.BORDER_CONSTANT, value=0)
    else:
        digit_padded = cleaned

    resized = cv2.resize(digit_padded, (28, 28), interpolation=cv2.INTER_AREA)
    return resized


def segment_digits(img: np.ndarray):
    """
    Segment multiple digits using contour detection.
    Returns list of (x, preprocessed_28x28) sorted left-to-right.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 50 or area > 50000:  # filter noise and too large
            continue
        if w < 5 or h < 5:  # too small
            continue
        roi = thresh[y:y+h, x:x+w]
        pad = int(max(w, h) * 0.2)
        roi_padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad,
                                         cv2.BORDER_CONSTANT, value=0)
        roi_resized = cv2.resize(roi_padded, (28, 28), interpolation=cv2.INTER_AREA)
        digits.append((x, roi_resized))

    digits.sort(key=lambda d: d[0])  # left to right
    return digits


def solve_equation(expression: str):
    """
    Safely evaluate a math expression string.
    Supports +, -, *, /, ^, (), integers, decimals.
    Returns (result, steps).
    """
    if not expression:
        return None, ["No expression provided."]

    # Sanitize
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)
    if not safe_expr.strip():
        return None, ["Could not parse expression."]

    steps = []
    try:
        steps.append(f"Expression: {expression}")
        result = eval(safe_expr, {"__builtins__": {}})  # restricted eval
        steps.append(f"Result: {result}")
        return result, steps
    except ZeroDivisionError:
        return None, ["Division by zero!"]
    except Exception as e:
        return None, [f"Error solving: {str(e)}"]


def parse_equation_from_text(text: str):
    """Extract math expression from OCR/NLP text."""
    # Normalize
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    # Find equation patterns like "3 + 4 = ?", "2 * 5", etc.
    patterns = [
        r"(\d+[\s]*[+\-*/÷×^][\s]*\d+(?:[\s]*[+\-*/÷×^][\s]*\d+)*)",
        r"(\d+[\s]*=[\s]*\d+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()
    return text.strip()


def generate_ai_feedback(expression: str, result, correct_result=None, child_name: str = "friend") -> dict:
    """
    Rule-based AI tutor feedback. Production would swap this for LLM API call.
    Generates child-friendly, encouraging feedback.
    """
    if result is None:
        return {
            "message": f"Hmm {child_name}, I couldn't read that. Let's try again! 🤔",
            "encouragement": "Every mistake is a step forward!",
            "hint": "Try writing the numbers more clearly.",
            "emoji": "🌟"
        }

    is_correct = (correct_result is None) or (abs(float(result) - float(correct_result)) < 0.001)

    if is_correct:
        messages = [
            f"Amazing work, {child_name}! You got it right! 🎉",
            f"Brilliant, {child_name}! {expression} = {result} ✅",
            f"Superstar! That's exactly right! ⭐",
        ]
        import random
        return {
            "message": random.choice(messages),
            "encouragement": "Keep up the fantastic work!",
            "hint": None,
            "emoji": "🏆",
            "correct": True
        }
    else:
        return {
            "message": f"Good try, {child_name}! The answer is {correct_result}, not {result}. Let's look at it together! 💪",
            "encouragement": "Mistakes help us learn!",
            "hint": f"Try breaking it into smaller steps: {expression}",
            "emoji": "💡",
            "correct": False
        }


def adaptive_next_level(performance: dict) -> dict:
    """
    Adaptive difficulty engine.
    Input: {accuracy: 0-1, avg_time_sec: float, streak: int, current_level: int}
    Output: {next_level: int, message: str}
    """
    acc = performance.get("accuracy", 0.5)
    streak = performance.get("streak", 0)
    level = performance.get("current_level", 1)

    if acc >= 0.85 and streak >= 3:
        next_level = min(level + 1, 10)
        msg = "You're doing great! Let's try something a bit harder! 🚀"
    elif acc < 0.5:
        next_level = max(level - 1, 1)
        msg = "Let's practice this a bit more first! You're doing well! 🌱"
    else:
        next_level = level
        msg = "Keep going! You're making great progress! ⭐"

    return {"next_level": next_level, "message": msg, "level_name": _level_name(next_level)}


def _level_name(level: int) -> str:
    names = {
        1: "Counting Stars ⭐",
        2: "Number Explorer 🔢",
        3: "Addition Adventure ➕",
        4: "Subtraction Safari ➖",
        5: "Multiplication Magic ✖️",
        6: "Division Dojo ➗",
        7: "Mixed Operations 🎯",
        8: "Word Problems 📖",
        9: "Mental Math Master 🧠",
        10: "Math Champion 🏆"
    }
    return names.get(level, f"Level {level}")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "DyscalculiaAI Platform is running 🚀"})


@app.route("/api/predict/digit", methods=["POST"])
def predict_digit():
    """
    Predict a single handwritten digit from canvas base64 image.
    Payload: { image: "data:image/png;base64,..." }
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image_b64(data["image"])
        preprocessed = preprocess_canvas(img)

        model = get_digit_model()
        digit, confidence, all_probs = model.predict(preprocessed)

        feedback = generate_ai_feedback(
            str(digit), digit,
            child_name=data.get("name", "friend")
        )

        logger.info(f"Single digit prediction: {digit} (conf={confidence:.2f})")
        return jsonify({
            "digit": int(digit),
            "confidence": round(float(confidence), 3),
            "probabilities": [round(float(p), 3) for p in all_probs],
            "feedback": feedback
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/multidigit", methods=["POST"])
def predict_multidigit():
    """
    Detect and predict multiple digits, then solve the equation.
    Payload: { image: "data:image/png;base64,...", solve: true }
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image_b64(data["image"])
        segments = segment_digits(img)

        if not segments:
            return jsonify({"error": "No digits detected", "digits": []}), 200

        model = get_digit_model()
        digits_result = []
        digit_string = ""

        for x_pos, seg in segments:
            digit, confidence, _ = model.predict(seg)
            digits_result.append({
                "digit": int(digit),
                "confidence": round(float(confidence), 3),
                "x_position": int(x_pos)
            })
            digit_string += str(digit)

        result_payload = {
            "digits": digits_result,
            "digit_string": digit_string
        }

        if data.get("solve", False) and digit_string:
            # Try to parse as equation (may contain +,-,*,/ from text layer)
            expr = data.get("expression") or digit_string
            result, steps = solve_equation(expr)
            result_payload["equation"] = expr
            result_payload["result"] = result
            result_payload["steps"] = steps
            result_payload["feedback"] = generate_ai_feedback(
                expr, result, child_name=data.get("name", "friend")
            )

        logger.info(f"Multi-digit result: {digit_string}")
        return jsonify(result_payload)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/solve", methods=["POST"])
def solve():
    """
    Solve a math equation from text.
    Payload: { expression: "3 + 4 * 2", name: "Alice" }
    """
    try:
        data = request.get_json()
        expression = data.get("expression", "").strip()
        if not expression:
            return jsonify({"error": "No expression provided"}), 400

        parsed = parse_equation_from_text(expression)
        result, steps = solve_equation(parsed)
        feedback = generate_ai_feedback(
            parsed, result, child_name=data.get("name", "friend")
        )

        return jsonify({
            "expression": parsed,
            "result": result,
            "steps": steps,
            "feedback": feedback
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/solve_equation", methods=["POST"])
def solve_equation_api():
    """
    Solve a typed equation.
    Payload: { equation: "12 + 3", name: "Alice" }
    """
    try:
        data = request.get_json()
        equation = data.get("equation", "").strip()
        if not equation:
            return jsonify({"error": "No equation provided"}), 400

        result, steps = solve_equation(equation)
        feedback = generate_ai_feedback(
            equation, result, child_name=data.get("name", "friend")
        )

        return jsonify({
            "equation": equation,
            "result": result,
            "steps": steps,
            "feedback": feedback
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/ocr", methods=["POST"])
def ocr_image():
    """
    Extract text/equations from uploaded image or PDF using Tesseract OCR.
    Accepts: multipart/form-data with file field 'image'
    """
    try:
        if "image" not in request.files:
            # Also accept base64 JSON
            data = request.get_json()
            if data and "image" in data:
                img = decode_image_b64(data["image"])
            else:
                return jsonify({"error": "No image provided"}), 400
        else:
            file = request.files["image"]
            img_bytes = file.read()
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Preprocess for OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, ocr_ready = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            import pytesseract
            pil_img = Image.fromarray(ocr_ready)
            raw_text = pytesseract.image_to_string(
                pil_img,
                config="--psm 6 -c tessedit_char_whitelist=0123456789+-*/=()^. "
            )
            text = raw_text.strip()
        except ImportError:
            text = "OCR unavailable (pytesseract not installed)"

        equation = parse_equation_from_text(text)
        result = None
        steps = []
        if equation:
            result, steps = solve_equation(equation)

        return jsonify({
            "raw_text": text,
            "equation": equation,
            "result": result,
            "steps": steps
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/speech", methods=["POST"])
def speech_to_text():
    """
    Convert speech audio to text using Wav2Vec2.
    Accepts: multipart/form-data with 'audio' WAV file
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()

        try:
            import torch
            import soundfile as sf

            audio_buffer = io.BytesIO(audio_bytes)
            audio_array, sample_rate = sf.read(audio_buffer)

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            processor, model = get_speech_pipeline()
            if processor is None:
                return jsonify({"error": "Speech model not loaded"}), 503

            inputs = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            equation = parse_equation_from_text(transcription.lower())
            result, steps = solve_equation(equation)

            return jsonify({
                "transcription": transcription,
                "equation": equation,
                "result": result,
                "steps": steps
            })

        except ImportError as ie:
            return jsonify({"error": f"Speech dependencies missing: {str(ie)}"}), 503

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/adaptive", methods=["POST"])
def adaptive():
    """
    Compute next difficulty level based on performance.
    Payload: { accuracy: 0.8, streak: 4, current_level: 3, avg_time_sec: 12 }
    """
    try:
        data = request.get_json()
        result = adaptive_next_level(data)
        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/problem/generate", methods=["POST"])
def generate_problem():
    """
    Generate a math problem for a given difficulty level.
    Payload: { level: 3, type: "addition" }
    """
    try:
        import random
        data = request.get_json() or {}
        level = data.get("level", 1)
        problem_type = data.get("type", "auto")

        # Level-based problem generation
        if level <= 2:
            a, b = random.randint(1, 9), random.randint(1, 9)
            expr, answer = f"{a} + {b}", a + b
            hint = f"Count {a} fingers, then add {b} more!"
        elif level <= 4:
            a, b = random.randint(10, 50), random.randint(1, 20)
            if problem_type == "subtraction" or (problem_type == "auto" and level == 4):
                expr, answer = f"{a} - {b}", a - b
                hint = f"Start at {a} and count back {b} steps."
            else:
                expr, answer = f"{a} + {b}", a + b
                hint = f"Try adding the tens first: {(a//10)*10} + {(b//10)*10}"
        elif level <= 6:
            a, b = random.randint(2, 12), random.randint(2, 12)
            if problem_type == "division" or (problem_type == "auto" and level == 6):
                product = a * b
                expr, answer = f"{product} ÷ {a}", b
                hint = f"Ask: {a} times what number equals {product}?"
            else:
                expr, answer = f"{a} × {b}", a * b
                hint = f"Think of {a} groups of {b} objects."
        else:
            a = random.randint(10, 99)
            b = random.randint(2, 9)
            c = random.randint(1, 20)
            expr = f"{a} + {b} × {c}"
            answer = a + b * c
            hint = "Remember: multiply before adding! (BODMAS)"

        return jsonify({
            "expression": expr,
            "answer": answer,
            "hint": hint,
            "level": level,
            "level_name": _level_name(level)
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting DyscalculiaAI Platform on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
