import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")
 
# 🛠️ Preprocessing Function
def preprocess_data(df, is_train=True, scaler=None, label_encoders=None, feature_names=None):
    df = df.drop(columns=["Patient ID", "Country", "Continent", "Hemisphere"], errors='ignore')

    if is_train:
        label_encoders = {}
        for col in ["Sex", "Diet"]:
            if col in df.columns:
                label_enc = LabelEncoder()
                df[col] = label_enc.fit_transform(df[col])
                label_encoders[col] = label_enc
        joblib.dump(label_encoders, "label_encoders.pkl")
    else:
        label_encoders = joblib.load("label_encoders.pkl")
        for col in ["Sex", "Diet"]:
            if col in df.columns and col in label_encoders:
                df[col] = df[col].map(
                    lambda x: label_encoders[col].transform([x])[0]
                    if x in label_encoders[col].classes_ else label_encoders[col].transform(["Average"])[0])

    if 'Blood Pressure' in df.columns:
        df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df.drop(columns=['Blood Pressure'], inplace=True)

    if is_train:
        X = df.drop(columns=["Heart Attack Risk"])
        y = df["Heart Attack Risk"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler, X.columns.tolist()
    else:
        X = df
        scaler = joblib.load("scaler.pkl")
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]
        X_scaled = scaler.transform(X)
        return X_scaled

# 🏥 Train Models
(train_test, scaler, feature_names) = preprocess_data(df)
joblib.dump(feature_names, "feature_names.pkl")
X_train, X_test, y_train, y_test = train_test

# ✅ Train Random Forest
rf_model = RandomForestClassifier(n_estimators=250, max_depth=20, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

# ✅ Train Deep Learning Model
dl_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
dl_model.save("dl_model.h5")

# ✅ Quantum Models
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

# 📊 Accuracy Calculation
print("\n📈 Model Accuracies:")

# Random Forest Accuracy
rf_test_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_test_pred)
print(f"🎯 Random Forest Accuracy: {rf_accuracy:.2%}")

# Deep Learning Accuracy
dl_test_pred = (dl_model.predict(X_test) > 0.5).astype("int32")
dl_accuracy = accuracy_score(y_test, dl_test_pred)
print(f"🤖 Deep Learning Accuracy: {dl_accuracy:.2%}")

# Quantum Neural Network Accuracy
qnn_preds = []
for sample in X_test[:, :3]:
    result = quantum_circuit(sample)
    qnn_preds.append(1 if result[0] < 0 else 0)
qnn_accuracy = accuracy_score(y_test[:len(qnn_preds)], qnn_preds)
print(f"⚛️ Quantum Neural Network Accuracy: {qnn_accuracy:.2%}")

# Quantum Machine Learning Accuracy
qml_preds = []
for _ in range(len(X_test[:20])):  # Limited for speed
    qml_out = qmn_circuit()
    prediction = interpret_qml_output(qml_out)
    qml_preds.append(prediction)
qml_accuracy = accuracy_score(y_test[:len(qml_preds)], qml_preds)
print(f"🔬 Quantum Machine Learning Accuracy (on 20 samples): {qml_accuracy:.2%}")

# 🔍 Prediction Function
def analyze_risk_factors(age, cholesterol, systolic_bp, diastolic_bp, smoking, exercise, diet):
    risk_factors = []
    if age > 50:
        risk_factors.append("Age above 50 increases heart attack risk.")
    if cholesterol > 200:
        risk_factors.append("High cholesterol levels (>200) contribute to artery blockages.")
    if systolic_bp > 135 or diastolic_bp > 90:
        risk_factors.append("High blood pressure strains the heart.")
    if smoking == 1:
        risk_factors.append("Smoking damages blood vessels and increases heart attack risk.")
    if exercise == 0:
        risk_factors.append("Lack of exercise increases heart disease risk.")
    if diet.lower() == "unhealthy":
        risk_factors.append("Unhealthy diet increases bad cholesterol and plaque buildup.")
    return "\n".join(risk_factors) if risk_factors else "The model detected risk based on complex patterns in your data."

# 🔮 Predict on New Data
def predict_new_data():
    print("\nEnter Patient Details:")
    sex = input("Sex (Male/Female): ")
    age = float(input("Age: "))
    cholesterol = float(input("Cholesterol Level: "))
    blood_pressure = input("Blood Pressure (e.g., 120/80): ")
    smoking = int(input("Smoking Habit (1: Yes, 0: No): "))
    exercise = int(input("Exercise (Hours per week): "))
    diet = input("Diet (Healthy/Unhealthy/Average): ")

    systolic_bp, diastolic_bp = map(float, blood_pressure.split('/'))
    new_data = pd.DataFrame([{
        "Sex": sex,
        "Age": age,
        "Cholesterol": cholesterol,
        "Systolic BP": systolic_bp,
        "Diastolic BP": diastolic_bp,
        "Smoking": smoking,
        "Exercise Hours Per Week": exercise,
        "Diet": diet
    }])

    for col in feature_names:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data_scaled = preprocess_data(new_data, is_train=False, feature_names=feature_names)

    rf_model = joblib.load("rf_model.pkl")
    dl_model = load_model("dl_model.h5")

    rf_pred = rf_model.predict(new_data_scaled)[0]
    dl_pred = int(dl_model.predict(new_data_scaled)[0][0] > 0.5)
    qnn_pred = quantum_circuit(new_data_scaled[0][:3])
    qnn_risk = 1 if qnn_pred[0] < 0 else 0
    qml_pred = qmn_circuit()
    qml_risk = interpret_qml_output(qml_pred)

    print(f"\n🏥 Prediction: {'Risk' if rf_pred else 'No Risk'} (Random Forest)")
    print(f"🚑 Deep Learning: {'Risk' if dl_pred else 'No Risk'}")
    print(f"⚛️ Quantum Neural Network: {'Risk' if qnn_risk else 'No Risk'}")
    print(f"🔬 Quantum Machine Learning: {'Risk' if qml_risk else 'No Risk'}")

predict_new_data()

print(f"\n✅ Final Summary of Accuracies:")
print(f" Random Forest Accuracy: {rf_accuracy:.2%}")
print(f" Deep Learning Accuracy: {dl_accuracy:.2%}")
print(f" Quantum Neural Network Accuracy: {qnn_accuracy:.2%}")
print(f" Quantum Machine Learning Accuracy: {qml_accuracy:.2%}")
