# nec-activity1

## 21.11.2025

- modified .gitignore to skip saving generated data files
- added L1 and L2 Regularisations to neuralnet_torch.py
- modified A1_3_MLR_and_Pytorch_Comparison.ipynb and added L0/L1/L2 parameter to fit (work in progress)

## 20.11.2025

**Update:** Updated data management

- The generated pre-processed datasets in notebook 1 are stored in the data folder and re-used by the other notebooks. This way, we are not re-generating datasets everytime a notebook is executed

## 18.11.2025

**Update:** Created new folder structure. data - models - notebook

- Under notebooks there are three documents:
  1. Pre processing
  2. Execution of hte BP from scratch
  3. Execution of BP PyTorch and MLR
     Disclaimer: Part of the pre process, such as scaling is repeated on the three notebooks. We plan to do a future refactor where we can reuse the scaled data in roder to repeat that part of the process on each execution.

# Neural Network Debugging & Improvement Log

# Date: 16 November 2025

---

1. Initial Issue: MLP error stuck around ~14–15

---

- The multi-layer network (tanh/sigmoid) showed no learning.
- Training error barely moved across epochs.
- Hyperparameter tuning (lr, momentum, hidden sizes) had no effect.

---

2. Hypothesis #1: Weight Initialization Too Large

---

- Original weights were in range (-1, 1).
- Caused activation saturation → gradients ~0.
- FIX: reduce initialization scale to (-0.1, 0.1).
- RESULT: no more overflow, but MLP still not learning.

---

3. Sanity Check: Train a Simple Linear Model

---

- Model: [61, 1], activation = linear.
- RESULT: trained perfectly (Val MSE ≈ 0.20).
- CONCLUSION: Backprop for single-layer networks works.

---

4. Sanity Check #2: Deep Network but Fully Linear

---

- Model: [61, 40, 15, 1], activation = linear.
- RESULT: learned correctly (Val MSE ≈ 0.35).
- CONCLUSION: Multi-layer backprop also works.
- PROBLEM ONLY HAPPENS WITH NON-LINEAR ACTIVATIONS.

---

5. Hypothesis #2: Wrong Output Activation for Regression

---

- Originally, output layer used same activation as hidden layers.
- This is incompatible with continuous regression targets.
- FIX: hidden layers use user-selected activation;
  output layer always uses linear activation.
- RESULT: still no improvement; MLP MSE remained ≈ 14.

---

6. Critical Discovery: Target `y` Was Not Scaled

---

- `cnt_log` ranges approx from 0 to 6.
- Non-linear activations (tanh/sigmoid) output in (-1, 1) or (0, 1).
- This mismatch caused:
  - output saturation,
  - vanishing gradients,
  - error plateau near 14.
- FIX: apply StandardScaler to the target y (train/val/test).
- Train model on scaled y, inverse-transform predictions afterward.

---

7. Result After Scaling Target `y`

---

MLP with non-linear activation (tanh) finally learned:
Train MSE: 0.279
Val MSE: 0.606
Test MSE: 0.704

Huge improvement from previous MSE ~14.

---

8. Additional Insight: Activation Functions

---

- After scaling y and fixing output activation:
  - tanh works well.
  - sigmoid also works correctly again.
- Problem was activation saturation caused by target scale mismatch.

---

9. Final Architecture Rules

---

- Hidden layers: tanh / sigmoid / relu allowed.
- Output layer: always linear for regression.
- Inputs: StandardScaler (already applied).
- Target y: StandardScaler (new and essential).

---

10. Final Conclusion

---

- Backpropagation implementation is correct.
- The key issues were:
  1. overly large weight initialization,
  2. wrong output activation for regression,
  3. missing target scaling.
- After fixes, MLP now learns and generalizes correctly.
- Ready for hyperparameter experiments and reporting.

## 13.11.2025

**Update:** Implemented Data Perapration (data-preparation.ipynb), loading encoded csv from A1_Test_Implementation.ipynb

- Converting ctn (count) to cnt_log, one-hot encoding categorical values
- Saving encoded dataset to csv
- Loading the CSV, splitting dataset using train_test_split

## 06.11.2025

**Update:** Finished BP from scratch version. Added Multilinear y BP with Pytorch, still in review.

### Implemented

- BP From scratch has the possibility of selecting the activation function
- Implemented the fit method
- Added new notebook called A1_Test_Implementations for executing a simple example and check functionality

### To check / next steps

- Verify scikit and pytorch implementation
- Skeleton added from official documentation, still in review

## 29.10.2025

**Update:** Started the development of the backpropagation method.

### Implemented

- Minimmum functionality OK
- Weight and bias initialization values changed to range from -1 to 1
- Notebook NeuralTest modified to try a fixed neural model and 5000 epochs

### To check / next steps

- Layear creation quantity is fixed, next change will be to add a method to allow the user to define number of layers.
- Verify calculations, specificly fro formulas 14 and 13 from the document [G]
- Develope a feature to select and decide which activation function the user wants to apply.

## 26.10.2025

**Update:** Created 'NeuralNet' class with initial structure of the neural network.  
Backpropagation and forward propagation are not coded yet.

### Implemented

- Variables initialization
- Dynamic structure of layers

### To check / next steps

- Activation functions (only sigmoid for now)
- Weight and bias initialization using values from '0' to '1'
  → Check if range should be '-1' to '1' in documentation
- Check and correct the code to be able to create adidtional layers or neurons.
