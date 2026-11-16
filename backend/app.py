from flask import Flask, request, jsonify
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Load the model pipeline
model = joblib.load('superkart_model.joblib')

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read CSV bytes from the file
        file_bytes = file.read()
        df_input = pd.read_csv(io.BytesIO(file_bytes))
        predictions = model.predict(df_input)

        # Format predictions as index: value mapping
        pred_dict = {str(i): float(val) for i, val in enumerate(predictions)}
        return jsonify(pred_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
