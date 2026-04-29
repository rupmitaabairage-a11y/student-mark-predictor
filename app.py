from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("student_mark_predictor.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    prediction = model.predict([[hours]])[0][0]
    result = round(prediction, 2)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)