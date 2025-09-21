from flask import Flask, request
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_cors import CORS   # 🔹 Add CORS support
import os
import csv

app = Flask(__name__)
CORS(app)  # 🔹 Enable CORS so Android/other clients can POST

# ==============================
#  CONFIGURATION
# ==============================

# 🔹 Base directory of your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Where feedback images & CSV will be stored
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'feedback_images')
CSV_FILE = os.path.join(BASE_DIR, 'feedback_records.csv')

# 🔹 Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 Tell Flask where to save uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
#  ROUTES
# ==============================

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    print("📩 Incoming Feedback")

    # Debug: see what the app sends
    print("Headers:", dict(request.headers))
    print("Form Data:", request.form.to_dict())
    print("Files:", request.files.to_dict())

    predicted = request.form.get('predicted') or ''
    correct = request.form.get('correct') or ''
    new_class = request.form.get('new_class') or ''

    file = request.files.get('image_file')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = ''
    if file:
        filename = secure_filename(timestamp + ".jpg")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Write to CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "predicted", "correct", "new_class"])
        writer.writerow([timestamp, filename, predicted, correct, new_class])

    return '✅ Feedback received', 200


if __name__ == '__main__':
    # Run on your local network so Android can connect
    app.run(host='0.0.0.0', port=5000, debug=True)
