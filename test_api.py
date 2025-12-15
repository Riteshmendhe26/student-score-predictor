import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "hours": 7,
    "attendance": 85,
    "previous_score": 70
}

response = requests.post(url, json=data)

print("Predicted Score:", response.json())
