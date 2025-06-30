# Heart Attack Risk Predictor

This is a Flask-based web application that predicts heart attack risk using four different models:
- **Machine Learning (Random Forest)**
- **Deep Learning (Keras)**
- **Quantum Neural Network (QNN)** using PennyLane
- **Quantum Machine Learning (QML)** using Qiskit

The app accepts key medical data like age, cholesterol, BP, and habits, and provides a prediction from each model. It also explains *why* the user may be at risk using interpretable medical reasoning.

![image](https://github.com/user-attachments/assets/1813a41c-1aeb-481f-be65-c0f19646deff)

## Features

-  Multi-model risk prediction: ML, DL, QNN, and QML
-  Input preprocessing using scalers and label encoders
-  **Explains why you may be at risk** with personalized analysis (e.g., "Smoking increases heart attack risk")
-  UI fully implemented using a **custom Figma design**
-  Clear interface with styled feedback for each model

## Figma Design

The entire frontend layout is based on a custom Figma prototype.  
It includes:
- Landing page
- Prediction input page
- Results page with a medical illustration and color-coded model outputs

HTML templates were implemented to match the exact layout, theme, and user flow.

---

## Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow / Keras
- **Quantum ML**: PennyLane, Qiskit
- **Frontend**: HTML + CSS (Jinja2 templates from Figma design)

---

## Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/pavani0410/heart-attack-risk-predictor.git
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

4. **Open in browser**
   ```
   http://127.0.0.1:5000/
   ```

---

## Input Format

- **Sex**: Male / Female
- **Age**: Integer
- **Cholesterol Level**: Numeric
- **Blood Pressure**: Systolic/Diastolic (e.g., 130/85)
- **Smoking Habit**: 1 (Yes) / 0 (No)
- **Exercise**: Hours per week
- **Diet**: Healthy / Unhealthy

---

## Output

- **Model predictions:**
  - Random Forest
  - Deep Learning
  - QNN
  - QML
- **Explanation of Risk:**
  - Highlights key contributing factors (e.g., age > 50, high BP, smoking, unhealthy diet)
  - Displayed in a styled alert box titled *"Why You May Be at Risk"*

---

## Project Structure

```
.
├── app.py
├── train.py
├── dataset.csv
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── result.html
├── static/
├── models/
│   ├── rf_model.pkl
│   ├── dl_model.h5
│   ├── dl_model.keras
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
└── assets/
    ├── a4fc8873-dcc0-4363-92ed-8767dbdedca9.png
    └── d43b0776-9a7b-4737-b486-4085bf030fc0.png
```

---

## License

This project is released under the **MIT License**.

## Credits
Heart Attack Risk Predictor module  developed and designed by Pavani Sharma, later integrated into this unified health risk prediction system.

