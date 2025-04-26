# Parallel and Distributed Computing Semester Project

## 📖 Project Title:
Optimized Machine Learning Pipeline for Binary Classification

---

## 💡 Objective:
Design and implement a **high-performance machine learning pipeline** for **binary classification**, using optimization techniques like:
- Parallel processing
- Distributed computing
- GPU acceleration

Target: **Maximize accuracy and minimize total processing time**.

---

## 🔍 Requirements Checklist:

| Task | Status |
|:-----|:------|
| Preprocess the data (handle missing values, encode categorical variables, normalize features) | ✅ Completed |
| Train a machine learning model (Logistic Regression, Random Forest, XGBoost) | ✅ Completed |
| Train a deep learning model (PyTorch Neural Network) | ✅ Completed |
| Evaluate (accuracy, confusion matrix, F1 score) | ✅ Completed |
| Measure and report total processing time | ✅ Completed |
| Parallel computing (e.g., multithreading, multiprocessing) | ❌ Not Done |
| Distributed computing (e.g., Dask, Spark, MPI) | ❌ Not Done |
| GPU acceleration (PyTorch on GPU) | ❌ Not Done |
| Compare CPU vs GPU / Parallel vs Serial setups | ❌ Not Done |
| Source code modular and well-commented | ✅ Partial (needs final cleaning) |
| Performance Report (Accuracy, Time, Resource usage) | ❌ Not Done |
| Presentation/Demo (Architecture, Approach, Key Findings) | ❌ Not Done |

---

## 🏋️ Work Done So Far:

1. **Data Preprocessing**
   - Label Encoding (`feature_5`)
   - One-Hot Encoding (`feature_3`)
   - Standardization (Scaler)

2. **Model Training**
   - Logistic Regression
   - Random Forest
   - XGBoost Classifier
   - PyTorch-based Neural Network

3. **Evaluation**
   - Accuracy
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)

4. **Execution Time**
   - Measured for each model separately

---

## 🔄 Next Steps:

- Implement **Parallel Computing** (e.g., using `joblib`, multiprocessing)
- Implement **Distributed Computing** (e.g., using Dask or Spark)
- **Train models on GPU** (PyTorch CUDA)
- Create a **Performance Report** (Comparisons, Graphs)
- Finalize **Source Code Documentation**
- Prepare **Presentation/Demo**

---

## 🏆 Deliverables:

- [ ] Source Code (Fully Modular + Commented)
- [ ] Performance Report (Accuracy, Time, Resource Usage)
- [ ] Presentation Slides

---

## ✨ Notes:
- Accuracy improvements have been achieved by moving from Logistic Regression to Random Forest, XGBoost, and finally to PyTorch Neural Network.
- Execution time has been measured but optimization (parallelism/GPU) is still pending.

---

# ⚡ Current Accuracy Snapshot:

| Model | Accuracy |
|:------|:---------|
| Logistic Regression | ~50-60% |
| Random Forest | ~70-80% |
| XGBoost | ~75-85% |
| Neural Network (PyTorch) | ~78-88% |

---

# 🚀 Let's push forward and complete the remaining tasks!
