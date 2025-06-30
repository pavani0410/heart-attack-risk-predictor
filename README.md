# Heart Attack Risk Predictor

This is a Flask-based web application that predicts heart attack risk using four different models:
- **Machine Learning (Random Forest)**
- **Deep Learning (Keras)**
- **Quantum Neural Network (QNN) using PennyLane**
- **Quantum Machine Learning (QML) using Qiskit**

It takes user health inputs and provides model-based predictions along with explanations of major risk factors.

---

## Features

- Multi-model heart attack risk prediction (ML, DL, QML, QNN)
- Preprocessing with scalers and label encoders
- Risk interpretation based on medical thresholds
- **Custom frontend built using Figma design**
- Flask web interface with form handling and dynamic result rendering

---

## Figma Design

The frontend interface of this application is based on a **custom UI design created in Figma**.  
It has been fully implemented using HTML templates to reflect the planned layout, color scheme, and user flow designed in Figma.

> This gives the application a clean, user-friendly, and consistent visual experience.

---

## Tech Stack

- **Backend**: Flask, Python
- **ML/DL**: scikit-learn, TensorFlow/Keras
- **Quantum**: PennyLane, Qiskit
- **Frontend**: HTML + CSS (based on custom Figma design)
- **Data**: joblib models, Keras `.h5`, and user form inputs

---

## File Structure

```
├── app.py                  # Flask app
├── train.py                # Training script
├── dataset.csv             # Dataset used for training
├── requirements.txt        # Project dependencies
├── templates/              # HTML pages (based on Figma UI)
├── static/                 # CSS/JS (if any)
├── models/
│   ├── rf_model.pkl
│   ├── dl_model.h5
│   ├── dl_model.keras
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
```

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-attack-risk-predictor.git
   cd heart-attack-risk-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Visit the app in browser**
   ```
   http://127.0.0.1:5000/
   ```

---

## Input Fields

- Sex (Male/Female)
- Age
- Cholesterol Level (mg/dL)
- Blood Pressure (Systolic/Diastolic)
- Smoking Habit (1 or 0)
- Exercise Hours Per Week
- Diet (Healthy/Unhealthy)

---

## Output

- Predictions from:
  - Random Forest (ML)
  - Deep Learning (Keras)
  - Quantum Neural Network (QNN)
  - Quantum ML using Qiskit (QML)
- Interpretable analysis of major health risk factors

---

## License

MIT License
