"""Support Vector Machine (SVM) model."""

import random
import numpy as np
from tqdm import tqdm

class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me

        batch_size  = len(X_train)
        gradient    = np.zeros((self.w.shape))
        
        for x, y in zip(X_train, y_train):
            '''for correct class y'''
            gradient[y, :] += self.w[y, :] * self.reg_const/batch_size
            for c in range(self.n_class):
                if np.dot(self.w[y, :], x) - np.dot(self.w[c, :], x) < 1 and c!=y:
                    gradient[y, :] -= x

            '''for incorrect class y!=c'''
            for c in range(self.n_class):
                if c!=y:
                    gradient[c, :] += self.w[c, :] * self.reg_const/batch_size
                    if np.dot(self.w[y, :], x) - np.dot(self.w[c, :], x) < 1:
                        gradient[c, :] += x

        if batch_size:
            return gradient/batch_size
        return gradient

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        y = y.reshape(-1, 1)
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0
    
        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        n_samples, n_features   = X_train.shape
        X_train                 = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        self.w                  = np.random.randn(self.n_class, n_features+1)

        for e in tqdm(range(self.epochs)):
            batch_size              = 600
            mini_batches            = self.create_mini_batches(X_train, y_train, batch_size)
            
            num_batches = len(mini_batches)

            X_mini, y_mini  = mini_batches[random.randrange(num_batches)]
            y_mini          = y_mini.astype(int)
            gradient        = self.calc_gradient(X_mini, y_mini)
            self.w          -= self.lr * gradient
        

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        y_pred = []
        for x in X_test:
            z   = np.dot(self.w, x)
            c   = np.argmax(z)
            y_pred.append(c)
        return np.array(y_pred)
