# 🩺 Disease Prediction & Clinical Risk Assessment Pipeline

> An end-to-end clinical machine learning pipeline built to estimate cardiovascular disease likelihood from structured patient data. Developed as part of the Agentic X machine learning internship program.

---

## 📌 Project Overview & Clinical Objective
In clinical healthcare applications, the cost of model errors is highly asymmetrical:
* **False Negatives (Type II Error):** Falsely classifying a diseased patient as healthy. **Consequence:** Catastrophic risk of untreated heart disease, myocardial infarction, or sudden cardiac death.
* **False Positives (Type I Error):** Falsely flagging a healthy patient as diseased. **Consequence:** Patient anxiety and requirement for secondary confirmatory screening (e.g., stress tests or CT angiograms).

This project focuses heavily on **minimizing False Negatives** by systematically evaluating multiple classifiers, tuning decision probability thresholds, and interpreting clinical confusion matrices.

---

## 🛠️ Tools & Tech Stack
* **Language & Core Libraries:** Python, Pandas, NumPy, Scikit-Learn
* **Model Training & Comparison:** Logistic Regression, Random Forest, Gradient Boosting
* **Evaluation Framework:** ROC-AUC, Precision, Recall (Sensitivity), Confusion Matrix Analysis
* **Environment:** Google Colab

---

## 📊 Dataset & Features Description
The project utilizes the **UCI Heart Disease dataset** (`heart_disease_uci.csv`), containing structured patient records with mixed numerical and categorical clinical attributes.

### Key Features:
* **`age`**: Age in years
* **`sex`**: Patient biological sex
* **`cp`**: Chest pain type (Typical angina, Atypical angina, Non-anginal, Asymptomatic)
* **`trestbps`**: Resting blood pressure (mm Hg)
* **`chol`**: Serum cholesterol (mg/dl)
* **`fbs`**: Fasting blood sugar > 120 mg/dl
* **`restecg`**: Resting electrocardiographic results
* **`thalch`**: Maximum heart rate achieved
* **`exang`**: Exercise-induced angina
* **`oldpeak`**: ST depression induced by exercise relative to rest
* **`slope`**: Slope of the peak exercise ST segment
* **`ca`**: Number of major vessels colored by fluoroscopy
* **`thal`**: Thalassemia type
* **`target`**: Binarized clinical diagnosis ($0$ = Healthy, $1$ = Heart Disease Present)

---

## 🤖 Models Evaluated & Performance Comparison
To establish rigorous benchmarks, three distinct classifiers were compared fairly using a stratified 80/20 train-test split:
1. **Logistic Regression:** Linear baseline providing straightforward statistical feature coefficients.
2. **Random Forest:** Bagging tree ensemble robust against outliers and non-linear interactions.
3. **Gradient Boosting:** Sequential boosting ensemble designed to iteratively minimize residual errors.

### Performance Summary (At Tuned Clinical Threshold = 0.35):
To prioritize patient safety and catch maximum cases, the decision threshold was lowered from the default $0.50$ to **$0.35$**.

| Model | ROC-AUC | Recall (Sensitivity) | Precision | False Negatives (Missed Cases) | False Positives (False Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.9035$ | $91.18\%$ | $78.81\%$ | $9$ | $25$ |
| **Random Forest** | **$0.9200$** | **$96.08\%$** | $77.78\%$ | **$4$** | $28$ |
| **Gradient Boosting** | $0.9116$ | $95.10\%$ | $78.86\%$ | $5$ | $26$ |

> **Winner:** **Random Forest** achieved the highest ROC-AUC ($0.9200$) and an exceptional **$96.08\%$ Recall**, missing only $4$ out of $102$ diseased patients in the test cohort.

---

## ⚠️ Model Limitations & Stated Use Restrictions

1. **Not for Autonomous Clinical Diagnosis:** This model outputs statistical risk probabilities based on historical tabular data and must never replace a licensed physician, clinical electrocardiogram (ECG), or invasive angiography.
2. **Not for Emergency Triage:** The dataset reflects stable multi-center cardiac screening profiles and is invalid for acute trauma or emergency room resuscitation.
3. **Treatment Prescriptions Prohibited:** Model outputs must never be used independently to prescribe pharmaceutical treatments or surgical interventions.

---

## 🚀 How to Run the Code
1. Open the `Disease_Prediction.ipynb` notebook in **Google Colab**.
2. Run the ingestion cell and upload your `heart_disease_uci.csv` file when prompted.
3. Execute the cells sequentially to reproduce preprocessing, model training, threshold tuning, and evaluation graphs.
