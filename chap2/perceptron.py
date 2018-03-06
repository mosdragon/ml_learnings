import numpy as np


class Perceptron(object):
    """
    Perceptron classifer:

    Params:
    :eta: float
        Learning rate
    :n_iter: int
        Passes over the training dataset
    :random_state: int
        Random number generator seed for random weight initialization
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.errors = []

    def dotprod(self, xi):
        """
        Return the dot product of the input with the bias added.
        """
        return np.dot(xi, self.weights[1:]) + self.weights[0]

    def predict(self, xi):
        """
        Return class label 1 if above 0
        Return class label -1 otherwise
        """
        return np.where(self.dotprod(xi) >= 0.0, 1, -1)

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # Add one additional weight bc of the bias
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=(1 + X.shape[1]))

        for idx in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * ( target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update

                # Count how many weights updates needed.
                errors += int(update != 0.0)

            self.errors.append(errors)

            if errors == 0.0:
                print("Reached convergence after {} iterations".format(idx+1))
                break

        return self.errors


class AdalinePerceptron(Perceptron):

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # Add one additional weight bc of the bias
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=(1 + X.shape[1]))

        self.costs = []

        for idx in range(self.n_iter):
            # Compute predictions for each instance.
            output = self.dotprod(X)

            # Get the difference between each true label and prediction.
            errors = (y - output)
                
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)

            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()

        return self.costs
