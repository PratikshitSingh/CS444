"""Perceptron model."""

import numpy as np
from tqdm import tqdm

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        n_samples, n_features = X_train.shape
        
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        self.w  = np.random.randn(self.n_class, n_features+1)

        for e in tqdm(range(self.epochs)):
            for i in range(n_samples):
                x, y    = X_train[i] , y_train[i]
                z       = np.dot(self.w, x)

                for c in range(self.n_class):
                    if z[c] > z[y] and c!=y:
                        self.w[c, :]    -= self.lr * x
                        self.w[y, :]    += self.lr * x
            
            self.lr = (1/(1+20*e)) * self.lr

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
