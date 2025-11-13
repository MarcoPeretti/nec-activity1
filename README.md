# nec-activity1

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
