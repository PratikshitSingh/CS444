"""Neural network model."""

from typing import Sequence

import numpy as np

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i] ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        self.opt = opt

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        assert X.shape[1] == W.shape[0], print('Error X.shape[1] != W.shape[0]')
        assert True, print('Error X.shape[1] != W.shape[0]')
        ret = X @ W + b
        #assert 0 , print('Deliberate error , {} , {}'.format(ret.shape, X.shape, b.shape))
        return ret

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        ret = np.maximum(0, X)
        #assert False, print('deliberate error {}'.format(ret[:20]))
        return ret

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        ret = 1 * ( X>0 )
        #ret = np.where( X>0, 1, 0)
        #assert False, print('deliberate error in relu_grad', ret[:10])
        return ret

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
      ret = np.where(x>=0 , 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
      return ret
    
    def sigmoid_grad(self, x:np.ndarray):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # check this here , if direct average is the right thing to do. 
      return np.mean( (y - p) ** 2 )

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each row of X
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # Subtracting the maximum value for numerical stability
        e_x = np.exp(X - np.max(X, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
      

    def softmax_grad(self, X):
        num_training_examples, num_classes = X.shape
        S = np.zeros((num_classes, num_classes))
        for i in range(num_training_examples):
            x_i = X[i].reshape(-1, 1)
            outer = np.dot(x_i, x_i.T)
            diag = np.diag(X[i])
            S += outer * (diag - X[i][:, np.newaxis])
        return S

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        #setting self.outputs X[0] 
        self.outputs['Z0'] = X
        self.outputs["X0"] = X
        for i in range(1, self.num_layers):
            
            #sequencially multiplying the layers and storing in the outputs dict.
            self.outputs["Z"+str(i)] =  self.linear(self.params['W' + str(i)] , self.outputs["X"+str(i-1)] , self.params["b"+str(i)]) 
            self.outputs["X"+str(i)] = self.relu(self.outputs["Z"+str(i)])
            
        # for last layer the activation is softmax 
        self.outputs["Z"+str(self.num_layers)] =  self.linear(self.params['W' + str(self.num_layers)] , self.outputs["X"+str(self.num_layers-1)] , self.params["b"+str(self.num_layers)]) 
        #self.outputs["X"+str(self.num_layers)] = self.softmax(self.outputs["X"+str(self.num_layers)])

        # setting the last layer as final y
        self.outputs["X"+str(self.num_layers)] = self.sigmoid (self.outputs["Z"+str(self.num_layers)] )

        # MSE loss cannot be used here , because we don't have the true labels here. 
        
        return self.outputs["X"+str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        for param in self.params: 
          self.gradients[param] = np.zeros(self.params[param].shape)
        self.Xgradients = {}
        if self.opt == 'Adam': 
           # intialize first moment, second moment, 
           # every paramter will have a seperate moment 
           self.moment_f = {}
           self.moment_s = {}
           for param in self.params:
              self.moment_f[param] = np.zeros(self.gradients[param].shape)
              self.moment_s[param] = np.zeros(self.gradients[param].shape)
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        
        #calculating loss
        loss = self.mse( self.outputs["X"+str(self.num_layers)] , y)
        #print('loss :', loss)
        # calculating the gradient of the last layer here becuase it is different. 
        
        self.Xgradients["Z" + str(self.num_layers)] = 2 * (self.outputs["X"+str(self.num_layers)] - y) * self.sigmoid_grad(self.outputs["Z"+str(self.num_layers)])/ (y.shape[0]*y.shape[1])
        #print(self.Xgradients["X" + str(self
        # num_layers)].shape)
        #print(self.outputs["X"+str(self.num_layers-1)][t].shape)
        self.gradients["W" + str(self.num_layers)] = self.outputs["X"+str(self.num_layers-1)].T @ self.Xgradients["Z"+str(self.num_layers)]
        self.gradients["b" + str(self.num_layers)] = np.sum(self.Xgradients["Z"+str(self.num_layers)] , axis =0)
        assert self.gradients["b" + str(self.num_layers)].shape == self.params["b" + str(self.num_layers)].shape
        
        i = self.num_layers -1 
        self.Xgradients["Z"+str(i)] = (self.Xgradients["Z"+str(i+1)] @ self.params["W"+str(i+1)].T) * self.relu_grad(self.outputs["Z"+str(i)]) 
        
        for i in range(self.num_layers-1, 0,-1): 
            # implement going backward in layers.
            self.gradients["W" + str(i)] = self.outputs["X"+str(i-1)].T @ self.Xgradients["Z"+str(i)]
            assert self.gradients["W" + str(i)].shape == self.params["W" + str(i)].shape

            self.gradients["b" + str(i)] = np.sum(self.Xgradients["Z"+str(i)], axis = 0)
            assert self.gradients["b" + str(i)].shape == self.params["b" + str(i)].shape

            self.Xgradients["Z"+str(i-1)] = (self.Xgradients["Z"+str(i)] @ self.params["W"+str(i)].T) * self.relu_grad(self.outputs["Z"+str(i-1)]) 
            assert self.Xgradients["Z"+str(i-1)].shape == self.outputs["Z"+str(i-1)].shape
        
        #self.update()
        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = 'SGD',
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if self.opt == "SGD": 
          for i in range(1, self.num_layers+1):
            #print('old W'+ str(i) , ": ", self.params["W" + str(i)] )
            assert self.params["W" + str(i)].shape == self.gradients["W" + str(i)].shape, print('Shape shift gradient')
            self.params["W" + str(i)] -= lr *  self.gradients["W" + str(i)]
            self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
        elif self.opt == "Adam": 
           for param in self.params: 
              self.moment_f[param] = b1 * self.moment_f[param] + (1-b1) * self.gradients[param]
              self.moment_s[param] = b2 * self.moment_s[param] + (1-b2) * self.gradients[param] **2
              assert self.moment_f[param].shape == self.params[param].shape
              self.params[param] -= self.moment_f[param]*lr/(np.sqrt(self.moment_s[param])+ eps)
        return
    
    def diagnose(self): 
      # print the shape of all the gradients 
      print('params are ' , self.params)
      for param in self.params: 
        print(param , '  :  ', self.params[param].shape)
      
      # print x outputs 
      for output in self.outputs: 
        print(output , '  :  ', self.outputs[output].shape)
