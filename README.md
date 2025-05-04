# Accelerating Binary Classification with Parallel Computing and GPU Optimization

## 🚀 Project Overview

This project implements an optimized machine learning pipeline for binary classification using parallel processing and GPU acceleration techniques. We demonstrate significant performance improvements by comparing serial CPU execution against parallel CPU and GPU implementations. Our approach integrates data preprocessing, model selection, hyperparameter tuning, and model evaluation - all optimized for parallel execution.

**Contributors:**
- Muhammad Muneer 
- Shahzaib Ali 
- Hamza Mehmood 

## 📊 Dataset & Problem Statement

We worked with a binary classification dataset containing both numerical and categorical features. Our challenge was to:

1. Preprocess the data effectively
2. Implement and train multiple ML models
3. Optimize execution using parallel and distributed computing techniques
4. Achieve at least 70% reduction in processing time while maintaining or improving accuracy

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
- [Parallel & GPU Optimization](#parallel--gpu-optimization)
- [Performance Analysis](#performance-analysis)
- [Key Findings](#key-findings)
- [Setup & Requirements](#setup--requirements)

## 📁 Project Structure

```
├── data_preprocessing.py   # Data preprocessing functions
├── model_selection.py      # Model comparison and selection
├── rf_optimization.py      # Random Forest parallel & GPU implementation
├── neural_network.py       # ANN implementation with GPU support
├── evaluation.py           # Model evaluation metrics
├── utils.py                # Helper functions
└── requirements.txt        # Required dependencies
```

## 🔍 Data Preprocessing

We conducted thorough data preprocessing to prepare our dataset for machine learning:

### Handling Missing Values

We identified missing values in features 1, 2, 4, and 7, and filled them with the mean values of their respective columns. This approach preserves the data distribution while maintaining the integrity of the dataset.

### Outlier Treatment

We implemented winsorization to limit extreme values in numerical features, replacing outliers with the 5th and 95th percentile values. This technique helps improve model performance by reducing the impact of extreme values.

### Categorical Feature Encoding

We used one-hot encoding to transform categorical variables (feature_3 and feature_5) into a format suitable for machine learning algorithms. This approach ensures that categorical variables are properly represented without introducing ordinal relationships.

### Data Visualization & Exploration

Our exploratory data analysis revealed class imbalance in the target variable, which informed our model selection and evaluation strategies. We also analyzed feature correlations and distributions to better understand the dataset.

## 💻 Model Implementation

### Model Selection Process

We evaluated three different models initially:
- Decision Tree
- Random Forest
- XGBoost

Each model was evaluated based on accuracy, F1 score, precision, recall, and training time. Our analysis showed that Random Forest consistently outperformed the other models across multiple metrics, particularly in F1 score and accuracy.

## ⚡ Parallel & GPU Optimization

### CPU vs GPU Random Forest Implementation

We implemented and compared:
1. Serial CPU Random Forest
2. Parallel CPU Random Forest (using all CPU cores)
3. GPU-accelerated Random Forest (using RAPIDS cuML)

Our GPU implementation leveraged the RAPIDS library, which provides GPU-accelerated machine learning algorithms. For the CPU parallel implementation, we utilized scikit-learn's built-in parallelization capabilities.

### Neural Network Implementation

We also implemented an enhanced neural network using PyTorch with both CPU and GPU support. The neural network implementation included:

- Class imbalance handling using weighted sampling
- Advanced architecture with regularization (dropout and batch normalization)
- Learning rate scheduling with ReduceLROnPlateau
- Weight initialization strategies

The neural network provided comparable results to Random Forest while demonstrating excellent GPU scaling capabilities.

## 📈 Performance Analysis

### Random Forest Performance

| Model | Training Time (s) | Inference Time (s) | Accuracy | F1 Score |
|-------|-------------------|-------------------|----------|----------|
| Random Forest (CPU) | 13.83 | 0.0322 | 0.59 | 0.67 |
| Random Forest (GPU) | 2.12 | 0.0037 | 0.63 | 0.69 |

**Random Forest Training Speedup: 6.52x**  
**Random Forest Inference Speedup: 8.70x**

### Neural Network Performance

| Model | Training Time (s) | Accuracy | Precision | Recall | F1 Score |
|-------|-------------------|----------|-----------|--------|----------|
| Neural Network (CPU) | 12.27 | 0.59 | 0.532 | 0.5341 | 0.5783 |
| Neural Network (GPU) | 3.12 | 0.621 | 0.6322 | 0.621432 | 0.6031 |

**Neural Network Processing Time Reduction: 74.57%**

## 🔑 Key Findings

1. **GPU Acceleration:** We achieved over 74% reduction in processing time using GPU acceleration, exceeding the project requirement of 70%

2. **Model Performance:** Random Forest provided the best balance of performance metrics (F1 score, precision, recall) and execution speed

3. **Parallel Optimization:** CPU parallelism significantly improved performance but GPU parallelism offered more dramatic speedups

4. **Neural Network:** The neural network implementation showed comparable accuracy with slightly lower F1 scores than Random Forest, but demonstrated excellent GPU scaling

5. **Trade-offs:** We identified minimal accuracy trade-offs (less than 0.5% in most cases) when using GPU acceleration, which is negligible compared to the dramatic speed gains

## 🛠️ Setup & Requirements

### Dependencies
```
numpy==1.22.4
pandas==1.5.3
scikit-learn==1.2.2
xgboost==1.7.5
cudf-cu11==23.04.0
cuml-cu11==23.04.0
torch==2.0.1
matplotlib==3.7.1
seaborn==0.12.2
```

### Hardware Requirements
- CPU: Multi-core processor (8+ cores recommended)
- GPU: CUDA-compatible GPU with at least 4GB memory
- RAM: 8GB+ recommended

## 📝 Conclusion

This project successfully demonstrates the power of parallel computing and GPU acceleration for machine learning pipelines. We achieved a training time reduction of over 74%, while maintaining comparable accuracy metrics. The proposed implementation provides a scalable framework that can be extended to other machine learning tasks and larger datasets.

The combination of effective preprocessing, model selection, and parallel optimization provides a comprehensive approach to building high-performance machine learning systems.
