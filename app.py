from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline
app = Flask(__name__)

pipeline = PredictPipeline()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form or any other source
        # For simplicity, assuming form fields are named 'bruises', 'odor', etc.
        input_data = [
            int(request.form['bruises']),
            int(request.form['odor']),
            int(request.form['gill_spacing']),
            int(request.form['gill_size']),
            int(request.form['gill_color']),
            int(request.form['stalk_surface_above']),
            int(request.form['stalk_surface_below']),
            int(request.form['ring_type']),
            int(request.form['spore']),
            int(request.form['population']),
            int(request.form['habitat'])
        ]

        # Use the predict method from the PredictPipeline
        result = pipeline.predict(input_data)
        return render_template("result.html" , result=result)

 
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=8080)