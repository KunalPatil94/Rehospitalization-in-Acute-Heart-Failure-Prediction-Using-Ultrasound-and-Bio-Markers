# 🫀 Early Rehospitalization Forecasting in Acute Heart Failure (AHF)
Application link : https://rehospitalization-in-acute-heart-failure.onrender.com/

> An AI-powered clinical decision support system that predicts 30-day rehospitalization risk in Acute Heart Failure patients using machine learning and point-of-care ultrasound indicators.

---

## 📌 Overview

Acute Heart Failure (AHF) is one of the leading causes of hospital readmissions, with approximately **25% of patients readmitted within 30 days** of discharge. Traditional discharge tools like NT-proBNP and body weight monitoring often fail to detect hidden fluid overload or hemodynamic instability.

This project bridges that gap by combining **clinical biomarkers** and **ultrasound-based parameters** with machine learning models to accurately identify high-risk patients before discharge — enabling early intervention and reducing preventable readmissions.

---

## 🎯 Key Features

- **Dual ML Model Architecture** — Logistic Regression (baseline) + XGBoost (advanced) with ensemble probability output
- **SMOTE Balancing** — Handles class imbalance in clinical datasets for more reliable predictions
- **Risk Stratification** — Classifies patients as Low / Moderate / High risk with probability scores
- **Role-Based Access Control** — Separate dashboards for Doctors, Nurses, and Admins with bcrypt-secured authentication
- **Real-Time Clinical Dashboard** — Streamlit interface for instant risk assessment during discharge decisions
- **Model Explainability** — Feature importance visualization (XGBoost) showing top clinical predictors
- **Model Performance Monitoring** — Live comparison of Accuracy, AUC, Sensitivity, and Specificity across models
- **Automated Alerts & Notifications** — Risk-based follow-up alerts for high-risk patients
- **Patient Data Management** — SQLite database storing assessments, predictions, and biomarker trends
- **PDF Report Generation** — Downloadable clinical reports for patient records

---

## 🧠 Clinical Input Features

| Feature | Description |
|---|---|
| NT-proBNP | Cardiac biomarker for heart stress |
| Discharge Body Weight | Weight at discharge (fluid overload indicator) |
| Lung B-Line Score | Ultrasound-based pulmonary congestion marker (>3 = high risk) |
| IVC Collapsibility Index | Inferior vena cava assessment via ultrasound |
| Ejection Fraction | Cardiac output percentage |
| Age, Gender, Demographics | Patient baseline data |
| Comorbidities | Diabetes, Hypertension, CKD, AFib flags |
| Creatinine, Systolic BP, Heart Rate | Vital clinical indicators |

> **Key Finding (Martins et al., 2025):** Lung B-line score >3 is the single strongest predictor of 30-day rehospitalization in AHF patients.

---

## 🤖 ML Models & Mathematical Foundation

### 1. Logistic Regression (Baseline)
```
p̂ = σ(wᵀx + b),   σ(z) = 1 / (1 + e⁻ᶻ)

Loss: L = -(1/N) Σ [ yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ) ] + λ||w||²₂
```
- Outputs interpretable probabilities
- Handles linear relationships between features and readmission risk
- L2 regularization to prevent overfitting

### 2. XGBoost (Advanced)
```
ŷᵢ = Σ fₖ(xᵢ),   fₖ ∈ F

Loss: L = Σ ℓ(yᵢ, ŷᵢ) + Σ (γT + ½λ Σ wⱼ²)
```
- Captures nonlinear relationships and feature interactions
- Handles missing/noisy clinical data effectively
- Provides feature importance for clinical insights
- Target AUC: ≥ 0.80

### Evaluation Metrics
`Accuracy` · `AUC-ROC` · `Sensitivity (Recall)` · `Specificity` · `Precision` · `F1-Score`

---

## 🏗️ Project Architecture

```
ahf_enhanced/
│
├── app.py                  # Main Streamlit application & UI
├── models.py               # Logistic Regression + XGBoost training & prediction
├── database.py             # SQLite patient records & prediction storage
├── auth.py                 # User authentication & role-based access (bcrypt)
├── data_generator.py       # Synthetic clinical data generation
├── data_validation.py      # Input validation & clinical range checks
├── explainability.py       # Feature importance & SHAP-style explanations
├── monitoring.py           # Model performance monitoring dashboard
├── alert_system.py         # Risk-based automated alerts
├── notifications.py        # Follow-up notification management
├── reporting.py            # PDF clinical report generation
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

---

## 🖥️ Application Screens

| Screen | Role | Description |
|---|---|---|
| User Registration / Login | All | Secure bcrypt-authenticated access |
| Risk Assessment Dashboard | Doctor / Nurse | Input patient data, get instant prediction |
| Biomarker Trends & Results | Doctor | View current and historical patient reports |
| Model Performance Monitor | Admin / Developer | Compare LR vs XGBoost metrics live |
| Feature Importance Chart | Admin / Developer | Top 10 XGBoost predictors visualization |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Models | Scikit-learn, XGBoost |
| Class Balancing | imbalanced-learn (SMOTE) |
| Frontend | Streamlit |
| Visualization | Plotly |
| Database | SQLite |
| Authentication | bcrypt |
| Reporting | PDF generation |
| Deployment | Streamlit Cloud |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/KunalPatil94/<repo-name>.git
cd ahf_enhanced

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---
https://rehospitalization-in-acute-heart-failure.onrender.com/

## 👥 User Roles

| Role | Access |
|---|---|
| **Doctor / Healthcare Professional** | Risk assessment, patient history, biomarker trends |
| **Nurse / Care Team** | Follow-up planning, patient alerts |
| **Admin / Developer** | Model monitoring, performance metrics, system management |

---

## 📊 Problem Statement

> AHF readmissions cost healthcare systems billions annually and significantly worsen patient outcomes. Current discharge protocols — relying solely on NT-proBNP and weight — miss subclinical congestion and subtle hemodynamic instability. This system provides **AI-driven, data-backed discharge decisions** by fusing clinical and ultrasound parameters into a single risk score.

---

## 📚 References

1. Martins et al. (2025) — *Lung and IVC POCUS, NT-Pro-BNP, and Discharge Body Weight as Predictors of Rehospitalization in AHF* — Journal of Clinical Medicine
2. Wang et al. (2021) — *Prognostic Value of Lung Ultrasound for Heart Failure Outcomes* — Arquivos Brasileiros de Cardiologia
3. Pang et al. (2021) — *BLUSHED-AHF: Lung Ultrasound-Guided ED Management* — JACC: Heart Failure
4. McDonagh et al. (2021) — *2021 ESC Guidelines for Diagnosis and Treatment of Acute and Chronic Heart Failure* — European Heart Journal

---

## 👨‍💻 Team

| Name | Role |
|---|---|
| Kunal R. Patil | Developer |
| Saurabh Gunjal | Developer |
| Animesh Keche | Developer |
| Yash Gadekar | Developer |

**Project Guide:** Prof. Smita S. Wagh

**Institution:** Jayawantrao Sawant College of Engineering, Hadapsar, Pune

**University:** Savitribai Phule Pune University | Academic Year 2025–26

---

## 📄 License

This project is developed for academic purposes under the Final Year Computer Engineering program at SPPU.

## 📄 License & Copyright

© 2026 Kunal R. Patil, Saurabh Gunjal, Animesh Keche, Yash Gadekar. All rights reserved.

This project and its source code are protected under copyright law. No part of this project —
including code, documentation, models, or design — may be reproduced, distributed, or used
for commercial purposes without explicit written permission from the authors.

Developed as a Final Year Project at Jayawantrao Sawant College of Engineering, Pune
under Savitribai Phule Pune University (Academic Year 2025–26).
