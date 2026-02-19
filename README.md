# ☀️ Solar Power Forecasting Using Machine Learning  
## Milestone 1 — Data Pipeline & Baseline Forecasting System

---

## 📌 Project Overview

This project develops a machine learning–based solar power forecasting system capable of predicting short-term solar energy generation using historical weather and power output data.

Milestone 1 establishes the foundational architecture of the system, including:

- Data processing pipeline  
- Feature engineering framework  
- Baseline predictive modeling  
- Interactive dashboard prototype  

Accurate solar forecasting is essential for:

- Grid stability  
- Energy scheduling  
- Renewable energy integration  
- System reliability  

---

## 🎯 Milestone 1 Objectives

- Build a structured solar data processing pipeline  
- Implement automated data validation and cleaning  
- Engineer time-aware and domain-relevant features  
- Train and evaluate a baseline forecasting model  
- Develop an end-to-end prediction workflow  
- Create an interactive visualization dashboard  

---

## 📊 Dataset Description

The dataset contains timestamped solar generation values along with weather variables influencing photovoltaic output.

### Key Features

- **Timestamp**
- **Solar Power Output**
- **Temperature**
- **Cloud Cover**
- **Humidity**
- **Wind Speed**

These variables capture atmospheric and temporal patterns that impact solar production.

---

## 🔍 Data Pipeline

### 1️⃣ Data Validation
- Required column verification  
- Timestamp parsing  
- Chronological sorting  
- Duplicate detection  
- Missing value analysis  

### 2️⃣ Data Cleaning
- Handling missing values  
- Standardizing formats  
- Structuring time-series data  

This ensures clean, consistent, and model-ready input data.

---

## ⚙️ Feature Engineering

Domain-aware and time-series features were engineered to improve predictive performance.

### Temporal Encoding
- Hour of Day  
- Day of Year  
- Encoded using sine and cosine transformations  

### Lag Features
- Previous power observations  

### Rolling Statistics
- Moving averages  
- Short-term variability indicators  

### Weather Interaction Features
- Temperature × Cloud interactions  

### Clear-Sky Proxy
- Daylight-based generation potential estimation  

---

## 🤖 Baseline Model

A tree-based ensemble regression model is used as the baseline forecasting approach.

### Model Outputs
- Predicted solar power  
- Residual error  

---

## 📏 Evaluation Metrics

Model performance is evaluated using:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- R² Score  

---

## 🔄 End-to-End Forecasting Pipeline

The forecasting workflow integrates:

1. Data validation  
2. Data cleaning  
3. Feature engineering  
4. Model prediction  
5. Result generation  

This ensures reproducible and automated forecasting from raw input to final output.

---

## 📊 Interactive Dashboard

A Streamlit-based dashboard prototype provides:

- Data upload functionality  
- Forecast visualization  
- Model performance metrics  
- Error analysis plots  
- Feature importance insights  

The dashboard demonstrates real-time usability and model interpretability.

---

## 🏗️ Project Structure

```
solar-forecasting/
│
├── data/
│   └── solar_data.csv
│
├── notebooks/
│   └── EDA_and_model.ipynb
│
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── pipeline.py
│
├── app/
│   └── streamlit_app.py
│
├── reports/
│   └── milestone_1_report.tex
│
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the Dashboard

```
streamlit run app/streamlit_app.py
```

---

## 📌 Milestone 1 Outcome

- Functional data pipeline  
- Engineered time-series features  
- Baseline forecasting model  
- Automated prediction workflow  
- Interactive dashboard prototype  

This milestone confirms the feasibility of automated solar power forecasting using machine learning.

---

## 🔮 Future Work (Milestone 2)

- Model optimization and comparison  
- Uncertainty quantification  
- Advanced explainability methods  
- Performance monitoring and drift detection  
- Real-time data integration  
- Production deployment scaling  

---

## 👨‍💻 Author

Your Name  
Course Name  
Institution Name  
