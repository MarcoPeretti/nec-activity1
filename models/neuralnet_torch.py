import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetTorch(nn.Module):
    """
    Neural network using PyTorch.



    Main ideas:
    - We build a sequence of Linear (fully connected) layers.
    - Between hidden layers we can add an activation function.
    - We use MSE as loss (because this is a regression problem).
    - We use SGD with momentum as optimizer.
    - PyTorch takes care of gradients and weight updates.
    """

    def __init__(self, n, fact='sigmoid', eta=0.1, alpha=0.9, epochs=1000, val_split=0.0):
        """
        Initialize the neural network.

        Parameters
        ----------
        n : list of int
            Number of neurons in each layer.
            Example: [input_dim, 40, 15, 1]

        fact : str
            Activation function name.
            Options: 'sigmoid', 'relu', 'tanh', 'linear'.

        eta : float
            Learning rate used by SGD.

        alpha : float
            Momentum coefficient for SGD.

        epochs : int
            Number of passes over the training data.

        val_split : float
            Fraction of the data used for validation.
            Example: 0.2 means 20% for validation.
        """

        # Call parent constructor (required for nn.Module)
        super(NeuralNetTorch, self).__init__()

        # Save configuration
        self.n = n
        self.fact = fact
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.val_split = val_split

        
        # Build the network architecture
       
        # We will create a "Sequential" model: a list of layers in order.
        layers = []

        for i in range(len(n) - 1):
            # Linear layer: connects layer i to layer i+1
            in_units = n[i]
            out_units = n[i + 1]
            layers.append(nn.Linear(in_units, out_units))

            # Add activation only between hidden layers (not after last layer)
            if i < len(n) - 2:
                if fact == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif fact == 'relu':
                    layers.append(nn.ReLU())
                elif fact == 'tanh':
                    layers.append(nn.Tanh())
                elif fact == 'linear':
                    # No activation, pure linear layer
                    pass

        # Wrap all layers into a single Sequential model
        self.model = nn.Sequential(*layers)

        # Loss function: mean squared error (regression)
        self.criterion = nn.MSELoss()

        # Optimizer: SGD with momentum
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.eta,
            momentum=self.alpha
        )

        # Lists for saving loss values over epochs
        self.train_errors = []
        self.val_errors = []

        # Small summary (useful when learning)
        print("NeuralNetTorch (PyTorch) initialized")
        print(" - Layers:", self.n)
        print(" - Activation:", self.fact)
        print(" - Learning rate:", self.eta, "| Momentum:", self.alpha)
        print(" - Epochs:", self.epochs, "| Val split:", self.val_split)

 
    # Forward pass
    
    def forward(self, x):
        """
        Forward pass through the network.

        x : input tensor of shape (n_samples, n_features)

        Returns:
        output tensor of shape (n_samples, n_outputs)
        """
        return self.model(x)

    
    # Training loop
    
    def fit(self, X, Z):
        """
        Train the neural network.

        High-level steps:
        1. Convert input data to PyTorch tensors.
        2. Optionally split into train and validation sets.
        3. For each epoch:
           - forward pass
           - compute loss
           - backpropagation
           - optimizer step (update weights)
           - store training and validation errors
        """

        # 1) Convert NumPy arrays to PyTorch tensors if necessary
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32)

        # Make sure targets have shape (n_samples, n_output)
        # so it matches the network output
        Z = Z.view(-1, self.n[-1])

        # 2) Create train / validation split (if val_split > 0)
        n_samples = X.shape[0]
        n_val = int(self.val_split * n_samples)

        if n_val > 0:
            # Random permutation of indices
            indices = torch.randperm(n_samples)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train, Z_train = X[train_idx], Z[train_idx]
            X_val,   Z_val   = X[val_idx],   Z[val_idx]
        else:
            # Use all data for training, no validation set
            X_train, Z_train = X, Z
            X_val,   Z_val   = None, None

        # Reset loss history
        self.train_errors = []
        self.val_errors = []

        # 3) Main training loop
        for epoch in range(self.epochs):

            # Reset gradients from previous step
            self.optimizer.zero_grad()

            # Forward pass on training data
            output = self.forward(X_train)

            # Compute loss (MSE)
            loss = self.criterion(output, Z_train)

            # Backpropagation: compute gradients
            loss.backward()

            # Update weights using the optimizer
            self.optimizer.step()

            # Save training loss for this epoch
            mse_train = loss.item()
            self.train_errors.append(mse_train)

            # If we have a validation set, evaluate it here
            if X_val is not None:
                with torch.no_grad():  # no gradients needed for validation
                    val_output = self.forward(X_val)
                    val_loss = self.criterion(val_output, Z_val)
                    mse_val = val_loss.item()
            else:
                mse_val = None

            self.val_errors.append(mse_val)

            # Print progress every 100 epochs (just for monitoring)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train MSE={mse_train:.6f}", end="")
                if mse_val is not None:
                    print(f" | Val MSE={mse_val:.6f}")
                else:
                    print()

   
    # Prediction
    
    def predict(self, X):
        """
        Compute predictions for new input data.

        X can be:
        - a NumPy array, or
        - a PyTorch tensor.

        Returns:
        - a NumPy array with shape (n_samples, output_dim)
        """

        # Convert NumPy to tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        # In prediction, we do not need gradients
        with torch.no_grad():
            output = self.forward(X)

        # Convert back to NumPy
        return output.numpy().reshape(-1, self.n[-1])

   
    # Loss history
    
    def loss_epochs(self):
        """
        Return the lists with train and validation loss values
        for each epoch. This is useful for plotting learning curves.
        """
        return self.train_errors, self.val_errors


# Documentation and resources used as reference for this implementation:
#
# - PyTorch Official Documentation — nn.Module:
#   https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# - PyTorch Official Tutorial — Neural Networks:
#   https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# - PyTorch Optimizers (SGD + momentum):
#   https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
#

