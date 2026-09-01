from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("models/xgb_fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    time = float(request.form["time"])
    amount = float(request.form["amount"])  

    # Create input with all 30 features
    input_data = [time] + [0] * 28 + [amount]

    input_df = pd.DataFrame(
        [input_data],
        columns=[
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
            "V8", "V9", "V10", "V11", "V12", "V13", "V14",
            "V15", "V16", "V17", "V18", "V19", "V20", "V21",
            "V22", "V23", "V24", "V25", "V26", "V27", "V28",
            "Amount"
        ]
    )

    # Scale Time and Amount
    input_df["Amount"] = scaler.transform(input_df[["Amount"]])
    input_df["Time"] = scaler.transform(input_df[["Time"]])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = "⚠️ Fraudulent Transaction"
    else:
        result = "✅ Normal Transaction"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)