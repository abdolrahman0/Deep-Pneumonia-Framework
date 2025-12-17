# Deep-Pneumonia-Framework: A Multi-Stage Triple-Validated ResNet Architecture

![AI in Healthcare](https://img.shields.io/badge/Domain-Medical%20Imaging-red)
![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![License-MIT](https://img.shields.io/badge/License-MIT-green)

## 🩺 Project Overview
This repository presents a high-precision Computer-Aided Diagnosis (CAD) system for automated pneumonia detection in chest X-ray images. Utilizing a fine-tuned **ResNet** backbone, the framework is designed to prioritize clinical safety through a rigorous **Triple-Validation Protocol**, ensuring robustness across heterogeneous data sources.

---

## 🔬 Evaluation Methodology & Data Integrity
To transcend standard accuracy metrics, this project addresses the **"Data Shift"** challenge by validating the model against three distinct environments:

### 1. Stage I: Internal Consistency (Keremberke Benchmark)
- **Objective:** Establish a performance baseline on a standardized dataset.
- **Results:** **95.20% Accuracy**.
- **Insights:** Proved the model's capacity for high-fidelity feature extraction.

### 2. Stage II: Cross-Domain Generalization (Pawlo2013 Stress Test)
- **Objective:** Evaluate the model against "Out-of-Distribution" data from independent clinical sources.
- **Results:** Achieved a critical **Sensitivity (Recall) of 96.88%**.
- **Conclusion:** Demonstrated resilience against technical variances in X-ray imaging (exposure, positioning, hardware).

### 3. Stage III: Final Verification (Yash/Kaggle Validation)
- **Objective:** Confirming long-term stability and cross-validation reliability.
- **Accuracy:** **94.16%** with an **F1-Score of 0.95**.

---

## 📊 Technical Performance Analytics

| Metric | Source Environment | Generalization Environment | Final Verification |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 95.20% | 88.57% | 94.16% |
| **Recall (Sensitivity)** | 95.20% | **96.88%** | 95.38% |
| **Precision** | 98.25% | 85.89% | 96.31% |
| **F1-Score** | 0.96 | 0.91 | 0.95 |

> **Critical Note on Bias:** The model exhibits a "Safety-First" conservative bias. In clinical settings, a higher Recall is prioritized over Precision to ensure zero-tolerance for False Negatives (missed diagnoses).

---

## 📖 Detailed Documentation
For an in-depth technical analysis, including confusion matrices, detailed error logs, and comparative visualization in Arabic, please refer to our full whitepaper on Notion:
[🔗 Comprehensive Project Whitepaper on Notion](رابط_النوشن_الخاص_بك_هنا)

---

## 🚀 Deployment & Inference
The model is hosted on **Hugging Face Spaces** for real-time inference.
[🌍 Live Demo on Hugging Face](رابط_الـ_Space_على_HuggingFace_هنا)

## ⚖️ Disclaimer
This tool is intended for research and educational purposes as a "Second Opinion" system. It should not replace professional medical judgment.
