# server.py
from flask import Flask, request, jsonify, send_from_directory
import os, csv, time

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")
FEEDBACK_IMG_DIR = os.path.join(BASE_DIR, "feedback_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "predicted", "correct", "new_class", "filename"])


@app.route("/feedback", methods=["POST"])
def feedback():
    """Receive feedback from Android app"""
    try:
        data = request.get_json()
        predicted = data.get("predicted", "")
        correct = data.get("correct", "")
        new_class = data.get("new_class", "")
        filename = data.get("filename", f"{int(time.time())}.jpg")

        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                    predicted, correct, new_class, filename])
        return jsonify({"status": "success", "message": "Feedback saved"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/latest-model", methods=["GET"])
def latest_model():
    """Serve the latest .tflite file to Android app"""
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".tflite")]
    if not files:
        return jsonify({"status": "error", "message": "No model yet"}), 404
    files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
    latest_file = files[0]
    return send_from_directory(MODELS_DIR, latest_file, as_attachment=True)


@app.route("/")
def home():
    return "✅ TrashLens Feedback Server running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
