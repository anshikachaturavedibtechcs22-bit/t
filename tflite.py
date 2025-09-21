import tensorflow as tf

# Input Keras model
keras_model = "Effi_WRM.keras"
tflite_model = "Effi_WRM.tflite"

# Load model
model = tf.keras.models.load_model(keras_model)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for mobile
tflite_model_data = converter.convert()

# Save model
with open(tflite_model, "wb") as f:
    f.write(tflite_model_data)

print(f"✅ Conversion done! Saved as {tflite_model}")
