# IdentiLLM: Classifying Large Language Models through Machine Learning

## Table of Contents

1. [Project Overview](#project-overview)
2. [Contributors](#contributors)
3. [Executive Summary](#executive-summary)
4. [Technical Setup](#technical-setup)
5. [Dataset Description](#dataset-description)
6. [Model Families Explored](#model-families-explored)
   - [Random Forests](#1-random-forests)
   - [Softmax Regression](#2-softmax-regression-multinomial-logistic-regression)
   - [Neural Networks](#3-neural-networks)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results and Performance Analysis](#results-and-performance-analysis)
9. [Tools and Technologies](#tools-and-technologies)
10. [Implementation Notes](#implementation-notes)
11. [Limitations and Future Work](#limitations-and-future-work)
12. [Conclusion](#conclusion)
13. [References](#references)

---

## Project Overview

IdentiLLM is a machine learning classification system designed to identify and distinguish between different Large Language Model (LLM) families based on student usage patterns and feedback. This project was developed as part of CSC311: Introduction to Machine Learning at the University of Toronto.

The primary objective of this project is to develop and evaluate machine learning models capable of classifying user experiences with three major LLM platforms: ChatGPT, Claude, and Gemini. By analyzing student responses regarding their interactions with these models, we aim to identify distinguishing characteristics that can reliably predict which LLM was being evaluated.

---

## Contributors

**Project Team:**

- Zhe Wang
- Virat Talan
- Yicheng Wang

---

## Executive Summary

This project implements and compares three machine learning model families for multi-class classification:

- **Random Forests**: Ensemble-based decision tree classifier
- **Softmax Regression**: Discriminative linear classifier for multi-class problems
- **Neural Networks**: Multi-layer perceptron with non-linear activation functions

**Best Performing Model:** Neural Networks (Test Accuracy: 67.4%)

The neural network model was selected as the final production model due to its consistent performance across different data splits and evaluation metrics. While Softmax Regression achieved slightly higher test accuracy (70.6%), the neural network demonstrated lower variance and more stable generalization to unseen data, making it the more reliable choice for deployment.

---

## Technical Setup

### Environment Requirements

- **Programming Language:** Python 3.x
- **Primary Libraries:**
  - scikit-learn (model implementation and evaluation)
  - pandas (data manipulation and preprocessing)
  - numpy (numerical computations)
  - matplotlib/seaborn (visualization)

### Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Project Structure

```
CSC311_Project/
├── README.md
├── training_data_clean.csv
├── project_baseline.py
├── pred_example.py
├── CSC311_Project.ipynb
└── docs/
```

---

## Dataset Description

### Dataset Overview

The dataset comprises student responses evaluating their experiences with three major Large Language Models (ChatGPT, Claude, and Gemini) in academic contexts. Each record represents a single student's evaluation of one specific LLM platform.

### Data Characteristics

**Feature Types:**

- **Quantitative Features:** Likert-scale responses (1-5) measuring:
  - Likelihood of model usage
  - Perceived helpfulness
  - Frequency of result verification
  - Occurrence of suboptimal or incorrect responses
- **Qualitative Features:** Open-ended text responses describing:
  - Model strengths (e.g., concept simplification, writing assistance)
  - Model weaknesses (e.g., citation accuracy, factual reliability)

**Class Distribution:** The dataset exhibits relatively balanced class distribution across the three LLM categories, making accuracy a suitable primary evaluation metric.

### Data Quality Issues

Several data quality challenges were identified during initial exploration:

1. **Missing Data:** Incomplete responses with uneven missingness patterns across labels
2. **Response Duplication:** Students frequently provided nearly identical answers across multiple entries, introducing potential data leakage risks
3. **Text Inconsistencies:** Irregular formatting, placeholder strings, and duplicated phrasing in open-ended responses

### Data Preprocessing Pipeline

**1. Data Cleaning**

- Loaded raw data into pandas DataFrame
- Identified null and invalid entries through systematic inspection
- Applied median imputation for missing Likert-scale features
- Utilized "no response" indicators for empty text fields

**2. Feature Engineering**

- **One-Hot Encoding:** Transformed categorical Likert-scale responses into binary columns for model compatibility
- **Text Vectorization:** Extracted frequent, relevant keywords from open-ended responses
- **Binary Feature Creation:** Converted keyword presence into binary features to minimize noise while preserving signal

**3. Data Splitting Strategy**

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%
- **Critical Constraint:** All three responses from each student were kept in the same subset to prevent data leakage

**4. Exploratory Data Analysis**

- Generated box plots and count plots for feature distribution analysis
- Confirmed class balance across LLM categories
- Identified features with high predictive potential through visual inspection

---

## Model Families Explored

### 1. Random Forests

**Rationale for Selection**

Random Forests were chosen for their ability to handle both categorical and continuous features while capturing complex, non-linear relationships in the data. The ensemble approach reduces variance and improves stability, making it particularly suitable for datasets with balanced class distributions.

**Implementation Details**

- **Framework:** scikit-learn's `RandomForestClassifier`
- **Optimization Strategy:** Ensemble-based learning without gradient descent
- **Feature Importance:** Provides interpretable feature importance rankings

**Hyperparameter Search Space**

| Hyperparameter      | Values Explored                      | Rationale                             |
| ------------------- | ------------------------------------ | ------------------------------------- |
| `criterion`         | {gini, entropy}                      | Compare impurity measures             |
| `n_estimators`      | {50, 100, 200, 300}                  | Balance ensemble size and computation |
| `max_depth`         | {None, 5, 8, 10, 13, 15, 20, 25, 30} | Control tree complexity               |
| `max_features`      | {sqrt, log2, None}                   | Feature subsampling strategies        |
| `min_samples_split` | {25, 30, 35, 40, 45}                 | Prevent overfitting                   |
| `min_samples_leaf`  | {10, 15, 20, 25, 30, 35, 40}         | Control leaf node size                |

**Validation Approach**

Grid search over hyperparameter combinations with validation set evaluation to identify optimal configuration balancing performance and generalization.

### 2. Softmax Regression (Multinomial Logistic Regression)

**Rationale for Selection**

Softmax Regression provides a discriminative linear classifier that excels on high-dimensional sparse text features while maintaining interpretability through direct feature-weight-to-class mappings.

**Implementation Details**

- **Framework:** scikit-learn's `LogisticRegression` with multinomial option
- **Solver:** L-BFGS optimizer for efficient multinomial logistic loss minimization
- **Regularization:** L2 penalty controlled by inverse parameter C
- **Convergence:** max_iter = 5000 with automatic early stopping

**Hyperparameter Search Space**

**Text Feature Extraction (TF-IDF):**
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `max_features` | {250, 500, 1000, 5000} | Vocabulary size control |
| `min_df` | {1, 2, 3, 4, 5} | Minimum document frequency |
| `ngram_range` | {(1,1), (1,2)} | Unigrams vs. unigrams + bigrams |

**Classifier:**
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `C` | {0.1, 0.5, 1.0, 5.0, 10.0} | Regularization strength |

**Validation Strategy**

All hyperparameter tuning performed exclusively on validation set. Test set reserved for final unbiased evaluation. Larger vocabularies showed higher training accuracy but reduced validation recall and F1, indicating overfitting.

### 3. Neural Networks

**Rationale for Selection**

Neural networks serve as universal function approximators capable of modeling complex, non-linear relationships between features and labels. The architecture automatically learns hierarchical feature representations, reducing the need for extensive manual feature engineering in high-dimensional feature spaces.

**Implementation Details**

- **Framework:** scikit-learn's `MLPClassifier`
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Activation Function:** ReLU (Rectified Linear Unit)
- **Regularization:** L2 penalty to prevent overfitting
- **Output Layer:** Softmax activation for multi-class classification
- **Mini-batch Size:** 60 samples (approximately 1/10 of training data)
- **Early Stopping:** Threshold of 300 gradient descent iterations

**Hyperparameter Search Space**

| Hyperparameter        | Values Explored                      | Rationale                             |
| --------------------- | ------------------------------------ | ------------------------------------- |
| **Hidden Layers**     | 1-3 layers                           | Balance capacity and overfitting risk |
| **Neurons per Layer** | {4, 8, 16, 32, 64, 128}              | Progressive capacity scaling          |
| **L2 Regularization** | {0.0001, 0.0005, 0.001, 0.005, 0.01} | Fine-grained regularization control   |
| **Learning Rate**     | {0.0001, 0.0005, 0.001, 0.005, 0.01} | Ensure stable convergence             |
| **Vocabulary Size**   | {5, 10, 25, 50, 100}                 | Limit text feature noise              |

**Feature Selection**

Applied mutual information-based feature selection to identify and retain only the most informative features, improving model efficiency and reducing overfitting on noisy features.

**Validation Approach**

Cross-validation with grid search over hyperparameter combinations. Validation accuracy guided hyperparameter selection. Student-level grouping maintained across all splits to prevent data leakage.

---

## Evaluation Metrics

All models were evaluated using the following metrics:

1. **Accuracy:** Primary metric due to balanced class distribution
2. **Precision:** Measures classification exactness (minimize false positives)
3. **Recall:** Measures classification completeness (minimize false negatives)
4. **F1 Score (Macro):** Harmonic mean of precision and recall, weighted equally across all classes

---

## Results and Performance Analysis

### Model Performance Comparison

| Model              | Test Accuracy | Precision | Recall    | F1 Score  |
| ------------------ | ------------- | --------- | --------- | --------- |
| Random Forest      | 65.8%         | 63.9%     | 64.2%     | 62.6%     |
| Softmax Regression | 70.6%         | 71.5%     | 70.7%     | 70.5%     |
| **Neural Network** | **67.4%**     | **67.3%** | **67.5%** | **66.8%** |

### Model Selection Rationale

Despite Softmax Regression achieving the highest test accuracy (70.6%), the Neural Network was selected as the final model for the following reasons:

1. **Performance Difference:** The 3-4% accuracy gap falls within expected random variation given the relatively small and noisy test set
2. **Statistical Significance:** Without formal statistical testing, the observed difference cannot be definitively attributed to superior model performance
3. **Consistency and Variance:** Neural Network demonstrated more stable performance across different random data splits, indicating lower variance and better generalization
4. **Production Reliability:** Lower variance makes the Neural Network more reliable for deployment on unseen data

**Expected Test Accuracy:** 67.4% (empirically validated across multiple random splits)

### Error Analysis

**Common Error Patterns:**

The confusion matrix analysis reveals systematic misclassification patterns across all three models:

**Softmax Regression Confusion Matrix:**

| Actual \ Predicted | ChatGPT | Claude | Gemini |
| ------------------ | ------- | ------ | ------ |
| **ChatGPT**        | 38      | 2      | 2      |
| **Claude**         | 3       | 28     | 11     |
| **Gemini**         | 6       | 13     | 23     |

**Key Observations:**

1. **Claude vs. Gemini Confusion:**

   - 13 Gemini responses misclassified as Claude
   - 11 Claude responses misclassified as Gemini
   - Significantly higher confusion rates compared to ChatGPT misclassifications

2. **ChatGPT Distinctiveness:**

   - ChatGPT shows clearer separation from other models
   - Lower false positive and false negative rates

3. **Root Cause Analysis:**
   - Generic responses (e.g., "I would use it to write code") provide insufficient model-specific signal
   - Short or ambiguous wording in open-ended responses
   - Similar usage patterns between Claude and Gemini in academic contexts

---

## Tools and Technologies

### Core Technologies

- **Python 3.x:** Primary programming language
- **scikit-learn:** Machine learning model implementation, training, and evaluation
- **pandas:** Data manipulation, cleaning, and preprocessing
- **numpy:** Numerical computation and array operations
- **matplotlib/seaborn:** Data visualization and exploratory analysis

### Development Tools

- **Jupyter Notebook:** Interactive development and experimentation
- **Git:** Version control (if applicable)

### Key scikit-learn Components

- `RandomForestClassifier`: Random Forest implementation
- `LogisticRegression`: Softmax Regression with L-BFGS solver
- `MLPClassifier`: Neural Network with SGD optimizer
- `TfidfVectorizer`: Text feature extraction
- `train_test_split`: Data partitioning utilities
- `GridSearchCV` (custom implementation): Hyperparameter optimization

---

## Implementation Notes

### Custom Preprocessing

Custom preprocessing code was developed to:

- Clean text fields and remove irrelevant artifacts (e.g., "THIS MODEL" placeholders)
- Handle missing entries with appropriate imputation strategies
- Normalize merged text columns for consistent feature extraction
- Remove stop words and perform basic text normalization

### Feature Engineering

- Implemented bag-of-words representation for text features
- Applied mutual information-based feature selection for neural networks
- Created one-hot encoded representations of categorical Likert-scale features

---

## Limitations and Future Work

### Current Limitations

1. **Dataset Size:** Relatively small dataset limits model capacity and generalization
2. **Feature Sparsity:** Text features remain sparse despite preprocessing efforts
3. **Claude-Gemini Separation:** Models struggle to distinguish between Claude and Gemini responses
4. **Statistical Validation:** Lack of formal significance testing for model comparison

### Future Improvements

1. **Data Augmentation:** Collect additional labeled data to improve model robustness
2. **Advanced Text Features:** Implement contextualized embeddings (e.g., BERT, word2vec)
3. **Ensemble Methods:** Combine predictions from multiple models for improved accuracy
4. **Class-Specific Features:** Engineer features specifically targeting Claude vs. Gemini distinction
5. **Cross-Validation:** Implement k-fold cross-validation for more robust performance estimates
6. **Hyperparameter Optimization:** Explore more sophisticated optimization techniques (e.g., Bayesian optimization)

---

## Conclusion

This project successfully implemented and evaluated three distinct machine learning model families for LLM classification based on user feedback. The Neural Network model, achieving 67.4% test accuracy with strong consistency across metrics, provides a reliable foundation for predicting LLM identity from student responses.

The analysis revealed that while ChatGPT responses exhibit distinctive characteristics, distinguishing between Claude and Gemini remains challenging due to similar usage patterns and generic response patterns. Future work should focus on collecting more diverse data and implementing advanced feature engineering techniques to improve inter-model discrimination.

---

## References

- scikit-learn Documentation: https://scikit-learn.org/
- CSC311 Course Materials, University of Toronto
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning

---

## License

This project was developed for academic purposes as part of CSC311 at the University of Toronto.

---

_Last Updated: Fall 2025_
