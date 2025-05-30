from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label encoder once
try:
    model = joblib.load("motor_health_model.pkl")
    label_encoder = joblib.load("motor_health_label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model or encoder: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Motor Health Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict_health():
    try:
        data = request.get_json()

        if not data or "sensor_data" not in data:
            return jsonify({"error": "Missing 'sensor_data' key in request body"}), 400

        sensor_data = np.array(data["sensor_data"])

        # Validate shape
        if sensor_data.ndim != 2 or sensor_data.shape[1] != 4:
            return jsonify({
                "error": "Each sensor data row must have exactly 4 values: [Voltage, Temperature, DeltaWaterLevel, MotorStatus]"
            }), 400

        # Make prediction
        predictions = model.predict(sensor_data)
        predicted_labels = label_encoder.inverse_transform(predictions)

        return jsonify({"predictions": predicted_labels.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use 0.0.0.0 for Railway hosting
    app.run(debug=True, host="0.0.0.0", port=5000)
