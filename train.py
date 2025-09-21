# -*- coding: utf-8 -*-
"""
train.py – retrain your Keras model from feedback_records.csv and generate updated model & .tflite
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# =============================
# Configuration
# =============================
FEEDBACK_FILE = "feedback_records.csv"          # CSV where feedback stored
MODEL_FILE = "Effi_WRM.keras"                   # base model
UPDATED_MODEL_FILE = "Effi_WRM_updated.keras"   # retrained model will be saved here
TFLITE_FILE = "Effi_WRM_updated.tflite"         # exported TFLite model
IMAGE_DIR = "feedback_images"

# Classes (must match your app)
class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# =============================
# Load Feedback Data
# =============================
if not os.path.exists(FEEDBACK_FILE):
    print("No feedback_records.csv found. Nothing to retrain.")
    exit()

df = pd.read_csv(FEEDBACK_FILE)

# Filter only incorrect + new_class provided
retrain_df = df[(df["correct"] == "No") & (df["new_class"].notna())].copy()

if retrain_df.empty:
    print("WARNING: No new incorrect feedback samples found. Nothing to retrain.")
    exit()

# =============================
# Load Model
# =============================
model_to_load_path = UPDATED_MODEL_FILE if os.path.exists(UPDATED_MODEL_FILE) else MODEL_FILE
if not os.path.exists(model_to_load_path):
    print(f"Base model {MODEL_FILE} not found. Cannot retrain.")
    exit()

print(f"Loading model from {model_to_load_path} ...")
model = load_model(model_to_load_path)

# =============================
# Prepare Training Data
# =============================
X_train, y_train = [], []
print("Preparing images...")
for i, row in enumerate(retrain_df.itertuples()):
    img_path = os.path.join(IMAGE_DIR, row.filename)
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(384, 384))
        img_array = img_to_array(img)
        X_train.append(img_array)
        y_train.append(row.new_class)
    else:
        print(f"Image not found: {img_path}")

if not X_train:
    print("WARNING: No valid image samples found for retraining.")
    exit()

X_train = np.array(X_train) / 255.0
y_indices = [class_names.index(lbl) for lbl in y_train]
y_train_cat = to_categorical(y_indices, num_classes=len(class_names))

# =============================
# Retrain Model
# =============================
print("Starting retraining...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_cat, epochs=10, batch_size=4, validation_split=0.2)

# Save updated model
print(f"Saving updated model to {UPDATED_MODEL_FILE} ...")
model.save(UPDATED_MODEL_FILE)

# =============================
# Convert to TFLite
# =============================
print(f"Converting {UPDATED_MODEL_FILE} to {TFLITE_FILE} ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

print(f"✅ Updated TFLite model saved as {TFLITE_FILE}")

print("All Done! – Retraining and TFLite export complete.")
