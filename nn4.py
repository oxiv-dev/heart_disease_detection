import numpy as np 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class FNN:
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes, threshold=None, class_weights=None, activation="relu"):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.threshold = threshold
        self.class_weights = class_weights
        self.activation = activation
        self.num_layers = len(hidden_layer_sizes) + 1
        self.layer_sizes = [num_inputs] + hidden_layer_sizes + [num_outputs]
        self.weights = [np.random.randn(input_size, output_size) * np.sqrt(2/input_size) 
                        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.biases = [np.zeros((1, output_size)) for output_size in self.layer_sizes[1:]]
        self.activations = {"relu": self.relu, "sigmoid": self.sigmoid, "softmax": self.softmax}

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        cache = []
        A = X
        for l in range(self.num_layers):
            Z = np.dot(A, self.weights[l]) + self.biases[l]
            if l == self.num_layers - 1:
                A = self.softmax(Z)
            else:
                A = self.activations[self.activation](Z)
            cache.append((A, Z))
        return A, cache

    def backward_propagation(self, X, y, y_pred, cache, learning_rate):
        print(f"y: {y.shape}")
        print(f"y_pred: {y_pred.shape}")
        dA_prev = y_pred - y
        print(f"dA: {dA_prev.shape}")
        for l in reversed(range(self.num_layers)):
            A, Z = cache[l]
            m = X.shape[0]
            if l == self.num_layers - 1:
                dZ = dA_prev
            else:
                dZ = dA_prev * (Z > 0)
            print(f"dZ: {dZ.shape}")
            print(f"A: {A.shape}")
            dW = 1/m * np.dot(dZ.T, A)
            db = 1/m * np.sum(dZ, axis=0, keepdims=True)
            print(f"dZ: {dZ.shape}")
            print(f"weights: {self.weights[l].T.shape}")
            dA_prev = np.dot(dZ.T, self.weights[l])
            print(f"db: {db.shape}")
            print(f"DW: {dW.shape}")
            print(f"weights: {self.weights[l].shape}")
            self.weights[l] = self.weights[l] - (learning_rate * dW)
            self.biases[l] -= learning_rate * db

    def categorical_cross_entropy(self, y_true, y_pred):
        v1 = np.log(y_pred)
        v2 = y_true
        v3 = np.squeeze(np.asarray(v2)) * np.squeeze(np.asarray(v1))
        v4 = -np.mean(v3)
        return v4

    def fit(self, X, y, num_epochs, learning_rate, batch_size):
        num_samples = X.shape[0]
        y = self.convert_to_prob_matrix(y)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_pred_batch, cache = self.forward_propagation(X_batch)
                epoch_loss += self.categorical_cross_entropy(y_batch, y_pred_batch)
                # y_pred_batch = np.argmax(y_pred_batch, axis=1)
                # y_batch = np.argmax(y_batch, axis=1)
                self.backward_propagation(X_batch, y_batch, y_pred_batch, cache, learning_rate)
            epoch_loss /= (num_samples/batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}: loss = {epoch_loss:.4f}")

    def convert_to_prob_matrix(self, target_classes):
        label_encoder = LabelEncoder()
        numerical_labels = label_encoder.fit_transform(target_classes)
        numerical_labels = numerical_labels.reshape(-1, 1)  # Reshape to a 2D array

        onehot_encoder = OneHotEncoder(categories='auto')
        onehot_encoder.fit(numerical_labels)
        prob_matrix = onehot_encoder.transform(numerical_labels).toarray()
        return prob_matrix

    def predict(self, X):
        y_pred, _ = self.forward_propagation(X)
        print(f"y_pred {y_pred}")
        if self.threshold is not None:
            y_pred = np.where(y_pred >= self.threshold, 1, 0)

        print(f"y_pred 2 {y_pred}")
        return y_pred
    
    def predict_proba(self, X):
        y_pred, _ = self.forward_propagation(X)
        return y_pred

# Example usage
# X = np.random.randn(100, 18)
# y = np.random.randint(0, 2, size=(100, 31))
# fnn = FNN(num_inputs=18, num_outputs=31, hidden_layer_sizes=[64, 32], threshold=0.5, activation="relu")
# fnn.train(X, y, num_epochs=50, learning_rate=0.001, batch_size=32)
# y_pred = fnn.predict(X)
# y_pred_proba = fnn.predict_proba(X)

