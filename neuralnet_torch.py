import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetTorch(nn.Module):
    """
    Implementation of a multilayer neural network using PyTorch.
    
    Using the PyTorch library for automatic differentiation,

    weight updates and training management.
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
            Activation function name ('sigmoid', 'relu', 'tanh', 'linear')
        eta : float
            Learning rate (η)
        alpha : float
            Momentum (α)
        epochs : int
            Number of epochs used during training
        val_split : float
            Percentage of data used for validation (between 0 and 1)
        """

        super(NeuralNetTorch, self).__init__()

        # Save configuration parameters
        self.n = n
        self.fact = fact
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.val_split = val_split

        
        # Architecture definition
        
        layers = []

        # Here dynamically create each fully connected layer based on n
        for i in range(len(n) - 1):
            layers.append(nn.Linear(n[i], n[i + 1]))  # fully connected layer

            # Add activation function between layers (except the last one)
            if i < len(n) - 2:
                if fact == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif fact == 'relu':
                    layers.append(nn.ReLU())
                elif fact == 'tanh':
                    layers.append(nn.Tanh())
                elif fact == 'linear':
                    pass  # no activation between layers

        # The complete model is the sequence of layers defined above
        self.model = nn.Sequential(*layers)

        # Mean Squared Error (MSE) as loss function
        self.criterion = nn.MSELoss()

        # Gradient Descent with momentum
        self.optimizer = optim.SGD(self.parameters(), lr=self.eta, momentum=self.alpha)

        # Print network configuration summary
        print("Neural network (PyTorch) has been initialized")
        print("Architecture (neurons per layer):", self.n)
        print("Activation function used:", self.fact)

    
    # Forward propagation
    
    def forward(self, x):
        """
        Perform forward propagation through the network.
        PyTorch automatically stores all gradients for backpropagation.
        """
        return self.model(x)

    
    # Training function (fit)
    
    def fit(self, X, Z):
        """
        Train the neural network using the given dataset.
        Divides data into training and validation sets (if val_split > 0)
        and trains during the defined number of epochs.
        """

        # Converting data to PyTorch tensors 
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            Z = torch.tensor(Z, dtype=torch.float32)

        # Split dataset into training and validation
        n_samples = X.shape[0]
        n_val = int(self.val_split * n_samples)
        indices = torch.randperm(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, Z_train = X[train_idx], Z[train_idx]
        X_val, Z_val = X[val_idx], Z[val_idx]

        # Lists to store errors for each epoch
        self.train_errors = []
        self.val_errors = []

        
        # Training loop
        
        for epoch in range(self.epochs):

            # Set gradients to zero before each iteration
            self.optimizer.zero_grad()

            # Forward propagation for training data
            output = self.forward(X_train)

            # Calculate loss (Mean Squared Error)
            loss = self.criterion(output, Z_train)

            # Backpropagation (automatic in PyTorch)
            loss.backward()

            # Update weights and biases according to the optimizer
            self.optimizer.step()

            # Compute training MSE for this epoch
            mse_train = loss.item()
            self.train_errors.append(mse_train)

            
            # Validation phase - test to review
            
            if n_val > 0:
                with torch.no_grad():  # disables gradient tracking
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

    #
    # Prediction
    
    def predict(self, X):
        """
        Compute network output for the given input data.
        Returns predictions as a NumPy array.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():  # no gradient tracking during prediction
            output = self.forward(X)
        return output.numpy()

    
    # Loss history
    
    def loss_epochs(self):
        """
        Return the training and validation loss evolution.
        Useful for plotting error vs epochs after training.
        """
        return self.train_errors, self.val_errors
