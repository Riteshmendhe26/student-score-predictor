from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("student_score_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    hours = data["hours"]
    attendance = data["attendance"]
    previous_score = data["previous_score"]

    prediction = model.predict([[hours, attendance, previous_score]])

    return jsonify({
        "predicted_score": prediction[0]
    })

if __name__ == "__main__":
    app.run()