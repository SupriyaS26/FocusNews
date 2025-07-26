from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import joblib
from flask_cors import CORS
import json
import subprocess
import os
app = Flask(__name__)
CORS(app)

model_x, model_y = None, None

@app.route('/upload', methods=['POST'])
def upload():
    global model_x, model_y
    data = request.json
    df = pd.DataFrame(data)
    X = df[['predX', 'predY', 'scrollY']]
    y_x = df['screenX']
    y_y = df['screenY']

    model_x = xgb.XGBRegressor().fit(X, y_x)
    model_y = xgb.XGBRegressor().fit(X, y_y)

    joblib.dump(model_x, 'x_model.pkl')
    joblib.dump(model_y, 'y_model.pkl')

    return jsonify({'message': 'Model trained', 'samples': len(df)})

@app.route('/predict', methods=['POST'])
def predict():
    global model_x, model_y
    if model_x is None or model_y is None:
        print("Loading models...")
        model_x = joblib.load('x_model.pkl')
        model_y = joblib.load('y_model.pkl')

    try:
        data = request.json
        print("Predict received:", data)

        # Validate input shape
        if not all(k in data for k in ['predX', 'predY', 'scrollY']):
            return jsonify({"error": "Missing keys"}), 400

        df = pd.DataFrame([data])
        print("Input DataFrame:", df)

        corrected_x = model_x.predict(df)[0]
        corrected_y = model_y.predict(df)[0]
        return jsonify({'x': float(corrected_x), 'y': float(corrected_y)})

    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/gaze_scores', methods=['POST'])
def receive_gaze_scores():
    try:
        scores = request.get_json(force=True)
        if not scores or not isinstance(scores, list):
            raise ValueError("Invalid or empty gaze score data")

        # DEBUG: print to verify what is coming in
        print("Received gaze scores:", scores)

        with open("gaze_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        print("gaze_scores.json written successfully.")
        return jsonify({"status": "received", "count": len(scores)})
    except Exception as e:
        print("Gaze score save failed:", str(e))
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/summarized', methods=['GET'])
def get_summarized_output():
    with open("output.json", "r",encoding="utf-8") as f:
        return jsonify(json.load(f))
@app.route('/run_pipeline', methods=['GET'])
def run_pipeline():
    try:
        subprocess.run(["python", "pipeline.py"], check=True)
        return jsonify({"status": "done"})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run()
