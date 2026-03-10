# 🩺 GlycoSense – Diabetes Risk Predictor

A Machine Learning based web application that predicts the likelihood of diabetes using patient health metrics.

This project uses a trained ML model to analyze medical features and provide a diabetes risk prediction. The goal is to demonstrate how machine learning can assist in early health risk detection.

---

# 📌 Project Overview

Diabetes is one of the most common chronic diseases worldwide. Early detection can help individuals manage their health and reduce complications.

**GlycoSense** uses a trained machine learning model to predict diabetes risk based on input features such as glucose level, BMI, age, and other medical indicators.

The project demonstrates the full ML workflow:

* Data preprocessing
* Model training
* Model serialization
* Web interface for predictions

---

# 🚀 Features

* Predict diabetes risk using ML
* User-friendly web interface
* Pre-trained machine learning model
* Real-time prediction from user inputs
* Clean and simple UI
* Deployable ML application

---

# 🧠 Machine Learning Model

The model was trained using the **Pima Indians Diabetes Dataset**, a commonly used dataset for diabetes prediction tasks.

# 🧾 Model Input Features

The machine learning model predicts diabetes risk based on the following health parameters:

| Feature                 | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| **Gender**              | Biological gender of the patient                                        |
| **Age**                 | Age of the individual                                                   |
| **Hypertension**        | Indicates whether the patient has high blood pressure (0 = No, 1 = Yes) |
| **Heart Disease**       | Indicates presence of heart disease (0 = No, 1 = Yes)                   |
| **Smoking History**     | Smoking status (never, former, current, etc.)                           |
| **BMI**                 | Body Mass Index — a measure of body fat based on height and weight      |
| **HbA1c Level**         | Average blood sugar level over the past 2–3 months                      |
| **Blood Glucose Level** | Current blood glucose concentration                                     |
| **Diabetes**            | Target variable (0 = Non-Diabetic, 1 = Diabetic)                        |

The model analyzes these features to estimate the likelihood of diabetes.

### Output

* **0 → Non-Diabetic**
* **1 → Diabetic Risk**

---

# 🛠 Tech Stack

## Programming

* Python

## Machine Learning

* Scikit-learn (Random Forest Classifier)
* Pickle (Model Serialization)

### Data Processing

* Pandas
* NumPy

### Data Storage

* CSV Files

### Interface

* HTML (basic frontend)

### Development Tools

* Git
* GitHub
* Visual Studio Code


# ⚙️ Installation

Clone the repository

```
git clone https://github.com/KUHUSHARMA10/GlycoSense-Diabetes-Risk-Predictor.git
```

Navigate into the project folder

```
cd GlycoSense-Diabetes-Risk-Predictor
```

Install dependencies

```
pip install -r requirements.txt
```

---


# 📊 Example Prediction Workflow

1. User enters health metrics
2. Data is processed
3. Model predicts diabetes risk
4. Result is displayed instantly

---

# 📷 Screenshots

[GlycoSense Web Interface](ui_preview.png.png)

# 👩‍💻 Author

**Kuhu Sharma**

Student | Machine Learning Enthusiast

GitHub:
https://github.com/KUHUSHARMA10

---

# 📜 License

This project is open-source and available under the MIT License.

---

⭐ If you like this project, consider giving it a star!
