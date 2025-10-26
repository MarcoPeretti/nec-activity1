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

    def __init__(self, n, fact='sigmoid'):
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
            Activation function name (default: 'sigmoid')
        """
      
        # Variables to define the base architecture of the neural network

        self.n = n                  # number of neurons per layer
        self.L = len(n)             # number of layers
        self.fact = fact            # activation function name 

        # Initialization of variables 

        self.h = [None] * self.L           # fields (h)
        self.xi = [None] * self.L          # activations (xi)
        self.w = [None] * (self.L - 1)     # weights (w)
        self.theta = [None] * (self.L - 1) # thresholds/bias (theta)
        self.delta = [None] * self.L       # error deltas (delta)

        # Arrays for the threshold update used in the learning phase

        self.d_w = [None] * (self.L - 1)
        self.d_theta = [None] * (self.L - 1)
        self.d_w_prev = [None] * (self.L - 1)
        self.d_theta_prev = [None] * (self.L - 1)

       
        # Weights and thresholds are randomly initilizated 


       
        for l in range(self.L - 1):
            n_in = n[l]
            n_out = n[l + 1]

            # Here random numbers are assigned in weights and biases, the random values are between 0 and 1
            # Bias is one value pero neuron, the number of wights depends on how manu neurons has the previous layer

            self.w[l] = np.random.uniform(0, 1, (n_out, n_in))
            self.theta[l] = np.random.uniform(0, 1, (n_out, 1))

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

    
    # Activation of Sigmoid function (g) and its derivative (g') - Later we'll define additional activation functions - Check with the professor
 
    def sigmoid(self, x):
        """Sigmoid activation: g(x) = 1 / (1 + e^{-x})"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid with respect to x (used in backpropagation)."""
        s = self.sigmoid(x)
        return s * (1 - s)

    
    # Forward propagation process - to review not checked yet - Rview calculations and checked against document
    
    def forward(self, x):
        """
        Perform forward propagation in the network.
        Stores the activations (xi) and fields (h) in corresponding lists.
        """
        # Convert input into a column vector
        self.xi[0] = x.reshape(-1, 1)

        # Loop through all layers (except input)
        for l in range(1, self.L):
            # h^(l) = w^(l) * xi^(l-1) - theta^(l)
            self.h[l] = np.dot(self.w[l - 1], self.xi[l - 1]) - self.theta[l - 1]

            # ξ^(l) = g(h^(l))  using sigmoid
            self.xi[l] = self.sigmoid(self.h[l])

        # The last activation is the network output
        return self.xi[-1]

   
    # Backpropagation - only skeleton for now, detailing calculations based on document 

    def backpropagation(self, x, y):
        """
        Skeleton of the backpropagation algorithm.
       
        """
        # Step 1: Forward pass
        output = self.forward(x)

        print("[Backpropagation skeleton] - TEST")
   

        # Return the current output so we can see predictions
        return output

    
    # Prediction - skeleton for the prediction
  
    def predict(self, x):
        """Compute network output."""
        return self.forward(x)
