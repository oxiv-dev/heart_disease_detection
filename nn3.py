import numpy as np

class FNN:
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.W1 = np.random.randn(num_inputs, hidden_layer_sizes[0]) * 0.01
        self.b1 = np.zeros((1, hidden_layer_sizes[0]))
        self.W2 = np.random.randn(hidden_layer_sizes[0], hidden_layer_sizes[1]) * 0.01
        self.b2 = np.zeros((1, hidden_layer_sizes[1]))
        self.W3 = np.random.randn(hidden_layer_sizes[1], num_outputs) * 0.01
        self.b3 = np.zeros((1, num_outputs))

    def softmax(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = np.maximum(0, Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        y_pred = self.softmax(Z3)
        cache = (X, Z1, A1, Z2, A2, Z3)
        return y_pred, cache

    def categorical_cross_entropy(self, y_true, y_pred):
        num_samples = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / num_samples
        return loss

    def backward_propagation(self, X, y_true, y_pred, cache, learning_rate):
        (X, Z1, A1, Z2, A2, Z3) = cache
        num_samples = X.shape[0]

        dZ3 = (y_pred - y_true) / num_samples
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (Z2 > 0)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update the weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def train(self, X_train, y_train, num_epochs=1000, learning_rate=0.01):
        for epoch in range(num_epochs):
            # Forward propagation
            y_pred, cache = self.forward_propagation(X_train)

            # Compute the loss
        loss = self.categorical_cross_entropy(y_train, y_pred)

        # Backward propagation
        self.backward_propagation(X_train, y_train, y_pred, cache, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} Loss: {loss:.4f}")

def predict(self, X):
        y_pred, _ = self.forward_propagation(X)
        if self.threshold is not None:
            y_pred = np.where(y_pred >= self.threshold, 1, 0)
        return y_pred
