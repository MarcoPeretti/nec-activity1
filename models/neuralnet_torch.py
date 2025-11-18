import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetTorch(nn.Module):
    """
    Simple implementation of a multilayer neural network using PyTorch.
    
    The network uses:
    - Linear (fully connected) layers
    - Optional activation functions between layers
    - MSE as the loss function
    - SGD with momentum for optimization

    This class follows the same ideas as the manual backprop network,
    but uses PyTorch for automatic differentiation and weight updates.
    """

    def __init__(self, n, fact='sigmoid', eta=0.1, alpha=0.9, epochs=1000, val_split=0.0):
        """
        Initialize the neural network structure and training settings.
        
        Parameters
        ----------
        n : list of int
            Defines the number of neurons in each layer.
            Example: [input_dim, 40, 15, 1]
            
        fact : str
            Activation function name ('sigmoid', 'relu', 'tanh', 'linear').
            
        eta : float
            Learning rate value used by the optimizer.
        
        alpha : float
            Momentum coefficient for SGD.
        
        epochs : int
            Number of training iterations.
        
        val_split : float
            Fraction of data to use for validation (0 = no validation).
        """

        super(NeuralNetTorch, self).__init__()

        # Store basic configuration parameters
        self.n = n
        self.fact = fact
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.val_split = val_split

        # -----------------------------
        # Build the network architecture
        # -----------------------------
        # We create a "Sequential" model: a list of layers executed in order.
        layers = []

        for i in range(len(n) - 1):
            # Fully connected layer
            layers.append(nn.Linear(n[i], n[i + 1]))

            # Add activation function between layers (except the final layer)
            if i < len(n) - 2:

                # Choose activation based on 'fact'
                if fact == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif fact == 'relu':
                    layers.append(nn.ReLU())
                elif fact == 'tanh':
                    layers.append(nn.Tanh())
                elif fact == 'linear':
                    # No activation added
                    pass

        # Store the full model as a PyTorch Sequential network
        self.model = nn.Sequential(*layers)

        # Loss function: Mean Squared Error for regression tasks
        self.criterion = nn.MSELoss()

        # Optimizer: Stochastic Gradient Descent with momentum
        self.optimizer = optim.SGD(self.parameters(), lr=self.eta, momentum=self.alpha)

        # Containers for storing loss evolution across epochs
        self.train_errors = []
        self.val_errors = []

        print("Neural network (PyTorch) initialized.")
        print("Layers:", self.n)
        print("Activation:", self.fact)

    # --------------------------
    # Forward pass of the network
    # --------------------------
    def forward(self, x):
        """
        Perform the forward pass through the network.
        PyTorch automatically builds the computation graph during this step.
        """
        return self.model(x)

    # --------------------------
    # Training loop
    # --------------------------
    def fit(self, X, Z):
        """
        Train the neural network using gradient descent.
        The training procedure:
        
        1. Convert data to PyTorch tensors.
        2. Split into training/validation sets (optional).
        3. Loop over epochs:
            - forward pass
            - compute loss
            - backward pass (automatic)
            - update weights
        """

        # Convert NumPy arrays into PyTorch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32)

        # Ensure output shape is (n_samples, n_output)
        Z = Z.view(-1, self.n[-1])

        # ----------------------------------
        # Optional validation set split
        # ----------------------------------
        n_samples = X.shape[0]
        n_val = int(self.val_split * n_samples)

        if n_val > 0:
            # Random permutation of sample indices
            indices = torch.randperm(n_samples)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train, Z_train = X[train_idx], Z[train_idx]
            X_val, Z_val = X[val_idx], Z[val_idx]
        else:
            # No validation set
            X_train, Z_train = X, Z
            X_val, Z_val = None, None

        # Reset loss history
        self.train_errors = []
        self.val_errors = []

        # ----------------------------------
        # Main training loop
        # ----------------------------------
        for epoch in range(self.epochs):

            # Reset accumulated gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.forward(X_train)

            # Compute MSE loss
            loss = self.criterion(output, Z_train)

            # Backpropagation (autograd)
            loss.backward()

            # Apply weight updates
            self.optimizer.step()

            # Record training error
            mse_train = loss.item()
            self.train_errors.append(mse_train)

            # Validate if required
            if X_val is not None:
                with torch.no_grad():
                    val_output = self.forward(X_val)
                    val_loss = self.criterion(val_output, Z_val)
                    mse_val = val_loss.item()
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

    # --------------------------
    # Prediction function
    # --------------------------
    def predict(self, X):
        """
        Compute predictions for new input data.
        Returns a NumPy array with shape (n_samples, output_dim).
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():  # No gradient calculation during inference
            output = self.forward(X)

        return output.numpy().reshape(-1, self.n[-1])

    # --------------------------
    # Loss history for plotting
    # --------------------------
    def loss_epochs(self):
        """
        Returns lists of training and validation MSE across epochs.
        Useful for plotting learning curves.
        """
        return self.train_errors, self.val_errors
    



    # Documentation and resources used as reference for this implementation:
#
# - PyTorch Official Documentation — nn.Module:
#   https://pytorch.org/docs/stable/generated/torch.nn.Module.html
#
# - PyTorch Official Tutorial — Neural Networks:
#   https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#
# - PyTorch Optimizers (SGD + momentum):
#   https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
#
# These resources were used as a base for structuring the forward pass,
# the training loop, and the use of autograd + optimizer updates.

