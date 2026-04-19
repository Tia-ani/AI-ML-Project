# 📉 Customer Churn Prediction and Agentic Retention Strategy System

## 👥 Team Members
- Harshit Jain  
- Vansh Dagar  
- Anishka Khurana  

---

## 🚀 Project Overview

Customer churn is a critical problem in the telecom industry, directly impacting revenue and customer lifetime value. While traditional machine learning models can identify customers at risk of leaving, they fail to provide **actionable insights** to retain them.

This project builds an **end-to-end intelligent system** that not only predicts churn risk but also generates **data-driven retention strategies** using Agentic AI.

The system integrates:
- 📊 Machine Learning (Churn Prediction)
- 🤖 LangGraph-based Agentic Workflow
- 🔍 Retrieval-Augmented Generation (RAG)
- 📈 Interactive Streamlit Dashboard

👉 The system transforms **prediction → decision → action**

---

## 🧠 Key Features

### 🔹 1. Churn Prediction (Machine Learning)
- Models evaluated:
  - Random Forest  
  - Gradient Boosting  
  - Logistic Regression  
- Best Model: **Gradient Boosting**
- Performance:
  - Cross-Validation ROC-AUC: **0.8453**
  - Test ROC-AUC: **0.8419**
- Handles class imbalance using **SMOTE**

---

### 🔹 2. Agentic AI Retention System
- Built using **LangGraph**
- Multi-step reasoning pipeline:
  - Factor Analysis (feature importance)
  - Context Retrieval (RAG)
  - Strategy Generation (LLM)

---

### 🔹 3. Retrieval-Augmented Generation (RAG)
- Embedding Model: `all-MiniLM-L6-v2`
- Vector Database: **ChromaDB**
- Chunking:
  - Size: 300
  - Overlap: 50
- Retrieves top **2 most relevant strategies**

---

### 🔹 4. Structured AI Output
Each generated report contains:
- Risk Summary  
- Contributing Factors  
- Recommended Actions  
- Business Disclaimer  

---

### 🔹 5. Streamlit Dashboard
- Individual customer prediction
- Batch dashboard (Extension feature)
- Risk gauge visualization
- Missing data handling & warnings
- Clean and interactive UI

---

## 🏗️ System Architecture
```bash
User Input
↓
Data Preprocessing & Imputation
↓
Gradient Boosting Model
↓
Churn Probability
↓
Threshold-Based Routing
├── Low Risk → Return prediction only
├── High Risk → LangGraph Agent
├── Analyze Factors
├── Retrieve Context (RAG)
├── Generate Strategy (LLM)
└── Structured Output
↓
Streamlit Dashboard
```
---

## 📁 Repository Structure
```bash
AI-ML-Project/
├── app.py # Main Streamlit app
├── requirements.txt
├── README.md
├── .gitignore
|
├── agents/
│ ├── retention_agent.py # LangGraph workflow
│ └── rag_setup.py # RAG pipeline
│
├── models/
│ ├── train_model.py # ML training pipeline
│ └── inference.py # Model inference logic
|
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│ └── retention_best_practices.md
│ └── clean_data.py
│ └── cleaned_data.csv
│
├── artifacts/
│ ├── model.pkl
│ ├── feature_importance.csv
│ ├── feature_names.json
│ ├── label_encoders.pkl
│ └── model_results.json
│
├── report/
│ └── main.tex # LaTeX report
```
---

## ⚙️ Setup Instructions
### 2. Create virtual environment

### 1. Clone the repository
```bash
git clone https://github.com/Tia-ani/AI-ML-Project.git
cd AI-ML-Project
```
### 2. Create virtual environmen

```bash

python -m venv venv
```
### 3. Activate environment
```bash
Mac/Linux:

source venv/bin/activate

Windows:

venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt

Create a .env file:
```
### 5. Run the application
```bash 
streamlit run app.py
```
---
