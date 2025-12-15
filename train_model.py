import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "Hours_Studied": [2, 4, 6, 8, 9],
    "Attendance": [60, 70, 80, 90, 95],
    "Previous_Score": [50, 55, 65, 75, 85],
    "Final_Score": [55, 60, 70, 85, 92]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Attendance", "Previous_Score"]]
y = df["Final_Score"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "student_score_model.pkl")

print("Model trained and saved successfully!")
