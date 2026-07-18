# Data Classification Using AI | Project 2

Welcome to the production repository for **Project 2: Data Classification Using AI**, completed as part of the DecodeLabs Industrial Training Program (Batch: 2026)[cite: 1]. This project demonstrates the step-by-step construction of an automated Supervised Learning pipeline using fundamental algorithmic logic to transform raw data measurements into intelligent decisions.

## 📋 Project Overview & Architecture
Rather than relying on static, hand-written heuristic rules, this application utilizes historical feature data to let a machine autonomously discover underlying decision boundaries. The architecture is structured cleanly across the industry-standard **Input ➔ Process ➔ Output (IPO)** framework.

### Technical Components:
* **Domain Dataset:** The classic Iris Flower Dataset consisting of 150 balanced samples spanning 3 biological classes (Setosa, Versicolor, Virginica) mapped across 4 spatial features (Sepal/Petal Lengths and Widths).
* **Data Gatekeeping:** Feature scaling via `StandardScaler` ensuring all raw data points normalize to a Mean = 0 and Variance = 1, eliminating feature magnitude bias.
* **Algorithmic Engine:** K-Nearest Neighbors (KNN) classification driven by proximity principles.
* **Hyperparameter Tuning:** Automated `GridSearchCV` paired with 5-Fold Cross-Validation to dynamically identify the absolute best $K$-value alongside spatial distance metrics.

---

## 💻 Code Structure & Implementation
The source pipeline is organized into three distinct, easily reviewable phases:

1. **Input:** Safe data ingestion and class balance checking.
2. **Process:** Order-bias data shuffling, stratified 80/20 train/test splitting, normalization, and optimization loops.
3. **Output:** Advanced multi-class evaluations generating a raw Confusion Matrix alongside precision, recall, and F1-score profiles.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required scientific computing libraries:
```bash
pip install numpy scikit-learn matplotlib

Execution
Run the optimized Python script directly via your terminal window:
python classification_pipeline.py
