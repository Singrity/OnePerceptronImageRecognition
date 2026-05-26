import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.n_inputs = n_inputs
        self.lr = lr
        self.weights = np.zeros(n_inputs, dtype=np.float64)
        self.bias = 0.0
        self.training_log = []

    def forward(self, x):
        # x - vec 28x28
        net = np.dot(self.weights, x) + self.bias
        out = 1 if net >= 0 else 0
        return net, out

    def predict_batch(self, X):
        net = X @ self.weights + self.bias
        out = (net >= 0).astype(int)
        return out, net

    def train(self, X, Y, max_epochs=50, error_threshold=0.0, ui_callback=None):
        self.training_log = []
        n = len(X)

        for epoch in range(max_epochs):
            total_error = 0
            misclassified = 0

            for i in range(n):
                net, out = self.forward(X[i])
                error = Y[i] - out # error {-1, 0, 1}
                total_error += abs(error)

                if error != 0:
                    misclassified += 1
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error

            self.training_log.append(
                {
                    "epoch": epoch,
                    "error": total_error,
                    "mis": misclassified,
                    "bias": self.bias,
                }
            )
            if ui_callback:
                ui_callback(epoch, total_error, misclassified, self.bias)
            if total_error <= error_threshold:
                break

        return self.training_log

