import os
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

model = joblib.load("student_score_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    hours = float(data["hours"])
    attendance = float(data["attendance"])
    previous_score = float(data["previous_score"])

    prediction = model.predict([[hours, attendance, previous_score]])

    return jsonify({
        "predicted_score": round(prediction[0], 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
