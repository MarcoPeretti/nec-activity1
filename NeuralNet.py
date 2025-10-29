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

    
    # Declaration of Sigmoid function (g) and its derivative (g') - Later we'll define additional activation functions - Check with the professor
 
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
        #Input to the vector with activation of the first layer and we use reshape to calculate the rows using one column

        self.xi[0] = x.reshape(-1, 1)   

        #Loop over the layers but avoiding first layer (input)

        for l in range(1, self.L):  

            #calculation of fields H following the formula n°8 from document [G]  (h^(l) = w^(l) * xi^(l-1) - theta^(l))

            self.h[l] = np.dot(self.w[l - 1], self.xi[l - 1]) - self.theta[l - 1] 
            
            #Applying and storing the results from the activation function, in this case is sigmoid. Function n°10 from document [G]  g(h)= 1/1+e^-h'

            self.xi[l] = self.sigmoid(self.h[l]) 

        # The last activation showed is the output layer

        return self.xi[-1]  

   

    

        # ----------------------------------------------------------
    # Backpropagation - Full implementation (Online with momentum)
    # ----------------------------------------------------------
    def backpropagation(self, x, y, eta=0.1, alpha=0.9):
        """
        Perform one iteration (one pattern) of the Online Backpropagation algorithm.
        
        Based on equations (11), (12), (14) from the document [G].
        
        Parameters
        ----------
        x : np.ndarray
            Input pattern (vector of shape (n_input,))
        y : np.ndarray
            Target output (vector of shape (n_output,))
        eta : float
            Learning rate (η) - controls how fast the weights are updated
        alpha : float
            Momentum (α) - helps smooth out oscillations in weight updates
        """

       
        # 1. Feed Forward
       
        # Compute network output for this pattern and store all activations (xi) and fields (h)
        output = self.forward(x)

       
        # 2. Delta Calculation of initial output vs. desired output  (Eq. 11)
        
      
        # We calculate the difference (delta) between the output from the epoch vs. the desired output. This calculation is based on formula n°11:  D^(L) = g'(h^(L)) * (o^(L) - z)
        # For this case, it's necesary to use the derivative of the sigmoid function theat we previously deeclared as sigmoid_derivative
        # Next step we need to check how to make dinamic the calculation of the layer, now is set as L - 1
        # Variable name should be y or z?

        self.delta[self.L - 1] = self.sigmoid_derivative(self.h[self.L - 1]) * (self.xi[self.L - 1] - y.reshape(-1, 1))

       
        # 3. Execution of the delta propagation through all the layers in reversed order
        
        # We need to define in which layer we start
        # The backwards propagation is done following the formula n°12 D^(l-1) = g'(h^(l-1)) * (D^(l) + W^(l))
        # Reversed loop: propagating deltas from last hidden layer back to the first hidden layer
        # Each delta[l] tells how much each hidden neuron contributed to the final error.

        
        for l in reversed(range(1, self.L - 1)): 
            
            self.delta[l] = self.sigmoid_derivative(self.h[l]) * np.dot(self.w[l].T, self.delta[l + 1])

       
        # 4. Update of the weight and bias for all the neurons 
       
        for l in range(self.L - 1):
            # Calculate the modificaction of the weight  with the momentum included
            # Here we follow the formula n°14
            self.d_w[l] = -eta * np.dot(self.delta[l + 1], self.xi[l].T) + alpha * self.d_w_prev[l]

            # Calculate the modification of the bias or thresholds
            # Here we follow the formula n°14
            self.d_theta[l] = eta * self.delta[l + 1] + alpha * self.d_theta_prev[l]

            # Updating here the weights and bias or thresholds 
            # Here we follow the formula n°15
            
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]

            # The changes are saved in  to use them in the next iteration
           

            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]

        
        # 5️. Compute and return the error 
        
        # Mean Squared Error (MSE) for this pattern
        error = 0.5 * np.sum((y.reshape(-1, 1) - output) ** 2)

        return error


    
    # Prediction - skeleton for the prediction
  
    def predict(self, x):
        """Compute network output."""
        return self.forward(x)
