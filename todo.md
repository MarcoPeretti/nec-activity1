# 🧠 Activity 1 — Prediction with Supervised Learning Models

## High-Level To-Do List (Sections 1 → 3.2)

---

### Part 1: Dataset Selection & Analysis

- [x] **Select a dataset** meeting mandatory criteria:
- [ ] **Document dataset details**:
  - [ ] Link to dataset source/webpage.
  - [ ] Description of features and target variable.
- [ ] **Perform preprocessing**:
  - [x] Handle missing values.
  - [x] Encode categorical variables.
  - [ ] Identify and treat outliers.
  - [ ] Normalize or transform variables (inputs/outputs as needed).
- [ ] **Generate and save** preprocessed/normalized files for later use.

---

### Part 2: Implement Neural Networks

- [x] **Implement Back-propagation neural network from scratch**

---

### Part 3: Prediction and Comparison of Models

#### 3.1 Hyperparameter Comparison and Selection

- [ ] Explore ≥ 10 different **hyperparameter combinations**:
  - Number of layers
  - Layer structure
  - Epochs
  - Learning rate
  - Momentum
  - Activation function
- [ ] For each combination:
  - [ ] Compute metrics: **MAPE**, **MAE**, **MSE**.
  - [ ] Record in a summary **table**.
  - [ ] Plot 2–3 **scatter plots** (predicted vs real values).
  - [ ] Plot 2–3 **error evolution plots** (training vs validation error).
- [ ] Write a **2–3 paragraph discussion** justifying chosen hyperparameters.

#### 3.2 Model Result Comparison

- [ ] Implement or use:
  - [ ] **Multiple Linear Regression (MLR-F)** from scikit-learn.
  - [ ] **Neural Network (BP-F)** using a library (TensorFlow/PyTorch/etc.).
- [ ] Compare all three models (BP, BP-F, MLR-F):
  - [ ] Describe parameters of each model.
  - [ ] Present a **comparison table** (MAPE, MAE, MSE).
  - [ ] Include **scatter plots** (predicted vs real) for each model.
  - [ ] Write **2 discussion paragraphs** comparing performance and functionality.

---

### Deliverables Summary

- [ ] PDF report with:
  - GitHub link
  - Dataset details & preprocessing explanation
  - Implementation overview
  - Metrics tables and plots
  - Discussions
- [ ] GitHub repo with:
  - Code for BP (from scratch)
  - Jupyter notebooks for BP-F and MLR-F comparisons
  - Preprocessed dataset files
