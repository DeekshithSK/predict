from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
import os
import csv
import joblib  


app = Flask(__name__)


MODEL_PATH = r"/Users/deekshithsk/Desktop/Paradox/Project_AR/Projectheart123/model_joblib_heart .pkl"  
model = joblib.load(MODEL_PATH)

CSV_FILE = "predictions.csv"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = [
            int(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["cp"]),
            int(request.form["trestbps"]),
            int(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["restecg"]),
            int(request.form["thalach"]),
            int(request.form["exang"]),
            float(request.form["oldpeak"]),
            int(request.form["slope"]),
            int(request.form["ca"]),
            int(request.form["thal"])
        ]


        input_data = np.array([data]) 
        print("Processed Input Data:", input_data)


        prediction = model.predict(input_data)[0]
        print("Model Prediction Output:", prediction) 


        prediction_text = "Heart Disease" if prediction == 1 else "No Heart Disease"


        df = pd.DataFrame([data], columns=[
            "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol",
            "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
            "Oldpeak", "ST Segment Slope", "Number of Major Vessels", "Thalassemia"
        ])
        df["Prediction"] = prediction_text
        df.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))

       
        return render_template('prediction.html', prediction_results=df.to_html(index=False), csv_filename=CSV_FILE)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 2001))
    app.run(host='0.0.0.0', port=port, debug=True)