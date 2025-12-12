from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os
import google.generativeai as genai

# Load the trained model
model = joblib.load('model.joblib')


app = Flask(__name__)

# HTML template for the input form and results display
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Air Quality Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 600px; margin: auto; }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        form div { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="number"] { width: calc(100% - 22px); padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 4px; border: 1px solid #ced4da; text-align: center; }
        .result p { margin: 0; font-size: 1.1em; color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Air Quality Prediction </h1>
        <form action="/predict" method="post">
            <div>
                <label for="co">CO(GT) (e.g., 2.6):</label>
                <input type="number" step="any" id="co" name="co" required>
            </div>
            <div>
                <label for="benzene">C6H6(GT) (e.g., 11.88):</label>
                <input type="number" step="any" id="benzene" name="benzene" required>
            </div>
            <div>
                <label for="nox">NOx(GT) (e.g., 166.0):</label>
                <input type="number" step="any" id="nox" name="nox" required>
            </div>
            <div>
                <label for="no2">NO2(GT) (e.g., 113.0):</label>
                <input type="number" step="any" id="no2" name="no2" required>
            </div>
            <div>
                <label for="temp">Temperature (T) (e.g., 13.6):</label>
                <input type="number" step="any" id="temp" name="temp" required>
            </div>
            <div>
                <label for="rh">Relative Humidity (RH) (e.g., 48.87):</label>
                <input type="number" step="any" id="rh" name="rh" required>
            </div>
            <div>
                <label for="ah">Absolute Humidity (AH) (e.g., 0.75):</label>
                <input type="number" step="any" id="ah" name="ah" required>
            </div>
            <button type="submit">Predict</button>
        </form>
        {% if prediction_text %}
            <div class="result">
                <p>{{ prediction_text | safe }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['co']),
        float(request.form['benzene']),
        float(request.form['nox']),
        float(request.form['no2']),
        float(request.form['temp']),
        float(request.form['rh']),
        float(request.form['ah'])
    ]

    final_features = np.array(features).reshape(1, -1)

    # Predict the numeric value
    prediction = model.predict(final_features)[0]

    # Severity categorization
    if prediction <= 1064.395:
        severity = "Low Severity"
    elif 1064.395 < prediction <= 1303.75:
        severity = "Medium Severity"
    else:
        severity = "High Severity"

   

    # Format output for web display
    output = (
        f"Predicted value: {prediction:.2f} "
        f"<br><strong>Severity Level:</strong> {severity}"
    )

    return render_template_string(HTML_TEMPLATE, prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
