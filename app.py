from flask import Flask, render_template, request
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

app = Flask(__name__)

# Load models & preprocessors
rf_model = joblib.load("rf_model.pkl")
dl_model = load_model("dl_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Extract feature names
try:
    feature_names = scaler.feature_names_in_
except AttributeError:
    try:
        feature_names = rf_model.feature_names_in_
    except AttributeError:
        feature_names = ["Sex", "Age", "Cholesterol", "Systolic BP", "Diastolic BP",
                         "Smoking", "Exercise Hours Per Week", "Diet"]

# Quantum models
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def quantum_circuit(inputs):
    qml.AngleEmbedding(inputs, wires=[0, 1, 2], rotation='Y')
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RY(0.5, wires=0)
    qml.RY(0.5, wires=1)
    qml.RY(0.5, wires=2)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

def qmn_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ry(0.5, 0)
    qc.ry(0.5, 1)
    qc.ry(0.5, 2)
    qc.measure_all()
    sampler = Sampler()
    result = sampler.run([qc]).result()
    return result.quasi_dists[0]

def interpret_qml_output(qml_pred):
    return 1 if sum(qml_pred.values()) > 0.5 else 0

# Risk analysis
def analyze_risk_factors(age, cholesterol, systolic_bp, diastolic_bp, smoking, exercise, diet):
    risk_factors = []
    if age > 50:
        risk_factors.append("Age above 50 increases heart attack risk.")
    if cholesterol > 200:
        risk_factors.append("High cholesterol (>200) contributes to blockages.")
    if systolic_bp > 135 or diastolic_bp > 90:
        risk_factors.append("High blood pressure strains the heart.")
    if smoking == 1:
        risk_factors.append("Smoking increases heart attack risk.")
    if exercise == 0:
        risk_factors.append("Lack of exercise increases risk.")
    if diet.lower() == "unhealthy":
        risk_factors.append("Unhealthy diet raises cholesterol and plaque buildup.")
    return "\n".join(risk_factors) if risk_factors else "Model detected risk from complex patterns."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    form = request.form

    user_data = pd.DataFrame([{
        "Sex": form["sex"],
        "Age": float(form["age"]),
        "Cholesterol Level": float(form["cholesterol"]),
        "Blood Pressure": form["blood_pressure"],
        "Smoking Habit": int(form["smoking"]),
        "Exercise": int(form["exercise"]),
        "Diet": form["diet"]
    }])

    rename_map = {
        "Cholesterol Level": "Cholesterol",
        "Smoking Habit": "Smoking",
        "Exercise": "Exercise Hours Per Week"
    }
    user_data.rename(columns=rename_map, inplace=True)

    user_data[["Systolic BP", "Diastolic BP"]] = user_data["Blood Pressure"].str.split("/", expand=True).astype(float)
    user_data.drop(columns=["Blood Pressure"], inplace=True)

    for col in ["Sex", "Diet"]:
        if col in label_encoders:
            user_data[col] = user_data[col].map(
                lambda x: label_encoders[col].transform([x])[0]
                if x in label_encoders[col].classes_
                else label_encoders[col].transform(["Average"])[0]
            )

    for col in feature_names:
        if col not in user_data.columns:
            user_data[col] = 0

    user_data = user_data[feature_names]
    X_scaled = scaler.transform(user_data)

    rf_pred = rf_model.predict(X_scaled)[0]
    dl_pred = int(dl_model.predict(X_scaled)[0][0] > 0.5)
    qnn_pred = quantum_circuit(X_scaled[0][:3])
    qnn_risk = 1 if qnn_pred[0] < 0 else 0
    qml_pred = qmn_circuit()
    qml_risk = interpret_qml_output(qml_pred)

    risk_factors = analyze_risk_factors(
        user_data["Age"][0], user_data["Cholesterol"][0],
        user_data["Systolic BP"][0], user_data["Diastolic BP"][0],
        user_data["Smoking"][0], user_data["Exercise Hours Per Week"][0],
        form["diet"]
    )

    return render_template("result.html",
                           rf_pred=rf_pred,
                           dl_pred=dl_pred,
                           qnn_pred=qnn_risk,
                           qml_pred=qml_risk,
                           risk_factors=risk_factors)

if __name__ == "__main__":
    app.run(debug=True)
