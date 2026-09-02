from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("models/xgb_fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load separate scalers
with open("models/amount_scaler.pkl", "rb") as file:
    amount_scaler = pickle.load(file)

with open("models/time_scaler.pkl", "rb") as file:
    time_scaler = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    time = float(request.form["time"])
    amount = float(request.form["amount"])

    # Create input data
    input_data = [time] + [0] * 28 + [amount]

    columns = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
        "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28",
        "Amount"
    ]

    input_df = pd.DataFrame([input_data], columns=columns)

    # Scale Time and Amount separately
    input_df["Amount"] = amount_scaler.transform(
        input_df[["Amount"]]
    )

    input_df["Time"] = time_scaler.transform(
        input_df[["Time"]]
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = "⚠️ Fraudulent Transaction"
    else:
        result = "✅ Normal Transaction"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)