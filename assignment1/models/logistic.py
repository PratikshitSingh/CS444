"""Logistic regression model."""

import copy
import numpy as np
from tqdm import tqdm

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w          = None
        self.lr         = lr
        self.epochs     = epochs
        self.threshold  = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        sigmoid = []
        for x in z:
            if x >= 0:
                w = np.exp(-x)
                sigmoid.append(1 / (1 + w))
            else:
                w = np.exp(x)
                sigmoid.append(w / (1 + w))
        return np.array(sigmoid)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # DONE: implement me
        n_samples, n_features = X_train.shape
        
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        self.w  = np.random.randn(1, n_features+1)

        y_train_copy = copy.deepcopy(y_train)
        for i in range(n_samples):
            if y_train_copy[i] == 0:
                y_train_copy[i] = -1
        
        for e in tqdm(range(self.epochs)):
            gradient        = np.zeros((1, n_features+1))
            for i in range(n_samples):
                x, y        = X_train[i].reshape(-1, 1), y_train_copy[i]
                z           = np.dot(self.w, x)

                gradient    += -np.dot(np.dot(y, (self.sigmoid(np.dot(-y, z)))), x.T)
            self.w  = self.w - self.lr * gradient/n_samples

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
            c   = self.sigmoid(z)
            cl  = 1 if c>=self.threshold else 0
            y_pred.append(cl)
        return np.array(y_pred)