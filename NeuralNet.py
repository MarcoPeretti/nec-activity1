import numpy as np


class NeuralNet:
    """
    Initial version of a multilayer neural network using the sigmoid activation formula.
    
    Variables detailed in the assignment:

    - L: number of layers
    - n: array with the number of neurons per layer
    - h: list of arrays for the fields (h)
    - xi: list of arrays for the activations (ξ)
    - w: list of matrices for the weights (w)
    - theta: list of arrays for the thresholds or bias (θ)
    - delta: list of arrays for the error propagation (Δ)
    - d_w: list of matrices for the weight changes (δw)
    - d_theta: list of arrays for the threshold/bias changes (δθ)
    - d_w_prev: list of matrices for previous weight changes (momentum)
    - d_theta_prev: list of arrays for previous threshold changes (momentum)
    - fact: name of the activation function (only 'sigmoid' for now)
    """

    def __init__(self, n, fact='sigmoid', eta=0.1, alpha=0.9, epochs=1000, val_split=0.0):
        """
        Initialization of the neural network.

        Parameters
        ----------
        n : list of int
            Number of neurons per layer. Example: [2, 3, 1]
            * n[0] = input layer size
            * n[-1] = output layer size
            * middle elements = hidden layers
        fact : str
            Activation function name (default: 'sigmoid', 'relu', 'tanh', 'linear')
        eta : float
            Learning rate (η) - default value if not specified during training
        alpha : float
            Momentum (α) - default value if not specified during training
        epochs : int
            Number of epochs used during training
        val_split : float
            Percentage of data used for validation (between 0 and 1)
        """

        # Variables to define the base architecture of the neural network
        self.n = n                  # number of neurons per layer
        self.L = len(n)             # number of layers
        self.fact = fact            # activation function name 
        self.eta = eta              # learning rate (η)
        self.alpha = alpha          # momentum (α)
        self.epochs = epochs        # number of epochs
        self.val_split = val_split  # percentage of validation data

        # Initialization of variables 
        self.h = [None] * self.L            # fields (h)
        self.xi = [None] * self.L           # activations (xi)
        self.w = [None] * (self.L - 1)      # weights (w)
        self.theta = [None] * (self.L - 1)  # thresholds/bias (theta)
        self.delta = [None] * self.L        # error deltas (delta)

        # Arrays for the threshold update used in the learning phase
        self.d_w = [None] * (self.L - 1)
        self.d_theta = [None] * (self.L - 1)
        self.d_w_prev = [None] * (self.L - 1)
        self.d_theta_prev = [None] * (self.L - 1)

        # Logic for activation function selection
        if fact == 'sigmoid':
            self.g = self.sigmoid
            self.g_deriv = self.sigmoid_derivative
        elif fact == 'relu':
            self.g = self.relu
            self.g_deriv = self.relu_derivative
        elif fact == 'tanh':
            self.g = self.tanh
            self.g_deriv = self.tanh_derivative
        elif fact == 'linear':
            self.g = self.linear
            self.g_deriv = self.linear_derivative
        else:
            raise ValueError(f"Unknown activation function '{fact}'. Supported: sigmoid, relu, tanh, linear")

        # Weights and thresholds are randomly initialized
        for l in range(self.L - 1):
            n_in = n[l]
            n_out = n[l + 1]

            # Bias is one value per neuron, the number of weights depends on how many neurons the previous layer has.
            # Here random numbers are assigned in weights and biases, the random values are between -1 and 1.
            self.w[l] = np.random.uniform(-1, 1, (n_out, n_in))
            self.theta[l] = np.random.uniform(-1, 1, (n_out, 1))

            # Gradients and momentum are set to 0, this will be updated during the back propagation
            self.d_w[l] = np.zeros((n_out, n_in))
            self.d_theta[l] = np.zeros((n_out, 1))
            self.d_w_prev[l] = np.zeros((n_out, n_in))
            self.d_theta_prev[l] = np.zeros((n_out, 1))

            # Checking if the matrices were created and updated correctly.

        print("Neural network has been initialized")
        print("Architecture (neurons per layer):", self.n)
        print("Activation function used:", self.fact)
        for l in range(self.L - 1):
            print(f" Layer {l+1}: w{self.w[l].shape}, theta{self.theta[l].shape}")

    # Declaration of activation functions and their derivatives
    def sigmoid(self, x):
        """Sigmoid activation: g(x) = 1 / (1 + e^{-x})"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        """ReLU activation: g(x) = max(0, x)"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2

    def linear(self, x):
        """Linear activation"""
        return x

    def linear_derivative(self, x):
        """Derivative of linear"""
        return np.ones_like(x)

    # Forward propagation process
    def forward(self, x):
        """
        Perform forward propagation in the network.
        Stores the activations (xi) and fields (h) in corresponding lists.
        """
        self.xi[0] = x.reshape(-1, 1)

        for l in range(1, self.L):
            # Each layer calculates its field using the activations of the previous one
            self.h[l] = np.dot(self.w[l - 1], self.xi[l - 1]) - self.theta[l - 1]
            # Apply activation function
            self.xi[l] = self.g(self.h[l])  # uses selected activation

        # Output of the last layer
        return self.xi[-1]

    # Backpropagation - Full implementation (Online with momentum)
    def backpropagation(self, x, z, eta=None, alpha=None):
        """
        Perform one iteration (one pattern) of the Online Backpropagation algorithm.
        Updates weights and biases after processing a single input pattern.
        """
        # Use default eta and alpha if not provided
        if eta is None:
            eta = self.eta
        if alpha is None:
            alpha = self.alpha

        # Forward pass to compute activations
        output = self.forward(x)

        # Compute delta for the output layer (Eq. 11)
        self.delta[self.L - 1] = self.g_deriv(self.h[self.L - 1]) * (self.xi[self.L - 1] - z.reshape(-1, 1))

        # Backward propagation of deltas through hidden layers (Eq. 12)
        for l in reversed(range(1, self.L - 1)):
            self.delta[l] = self.g_deriv(self.h[l]) * np.dot(self.w[l].T, self.delta[l + 1])

        # Weight and bias updates (Eqs. 14 & 15)
        for l in range(self.L - 1):
            self.d_w[l] = -eta * np.dot(self.delta[l + 1], self.xi[l].T) + alpha * self.d_w_prev[l]
            self.d_theta[l] = eta * self.delta[l + 1] + alpha * self.d_theta_prev[l]
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]

        # Mean Squared Error for this sample
        error = 0.5 * np.sum((z.reshape(-1, 1) - output) ** 2)
        return error

    # Prediction
    def predict(self, x):
        """Compute network output for a given input."""
        return self.forward(x)

    # Fit method to train the network for multiple epochs
    def fit(self, X, Z):
        """
        Train the neural network using all input patterns (X) and targets (Z)
        for the number of epochs defined in the constructor.
        Divides the data into training and validation sets (if val_split > 0),
        performs forward and backward propagation, and stores the MSE evolution.
        """
        n_samples = X.shape[0]

        # Split dataset into training and validation
        n_val = int(self.val_split * n_samples)
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, Z_train = X[train_idx], Z[train_idx]
        X_val, Z_val = X[val_idx], Z[val_idx]

        # Lists to store MSE evolution during epochs
        self.train_errors = []
        self.val_errors = []

        # Training loop over all epochs
        for epoch in range(self.epochs):
            total_error = 0

            # Online update: backpropagation for each training sample
            for x, z in zip(X_train, Z_train):
                total_error += self.backpropagation(x, z)

            # Compute mean squared error (MSE) for training set
            mse_train = total_error / len(X_train)
            self.train_errors.append(mse_train)

            # If validation data exists, compute its MSE too
            if n_val > 0:
                total_val_error = 0
                for x, z in zip(X_val, Z_val):
                    output = self.forward(x)
                    total_val_error += 0.5 * np.sum((z.reshape(-1, 1) - output) ** 2)
                mse_val = total_val_error / len(X_val)
            else:
                mse_val = None

            self.val_errors.append(mse_val)

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train MSE={mse_train:.6f}", end="")
                if mse_val is not None:
                    print(f" | Val MSE={mse_val:.6f}")
                else:
                    print()

    # Return the stored loss evolution
    def loss_epochs(self):
        """
        Returns the training and validation loss evolution.
        Useful for plotting the error vs. epochs curve after training.
        """
        return self.train_errors, self.val_errors
