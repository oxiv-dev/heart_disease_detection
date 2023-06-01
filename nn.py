import numpy as np

# Define the architecture of the FNN
n_inputs = 18
n_hidden1 = 64
n_hidden2 = 128
n_hidden3 = 128
n_outputs = 11

# Initialize the weights and biases
weights = {
    'hidden1': np.random.randn(n_inputs, n_hidden1),
    'hidden2': np.random.randn(n_hidden1, n_hidden2),
    'hidden3': np.random.randn(n_hidden2, n_hidden3),
    'output': np.random.randn(n_hidden3, n_outputs)
}

biases = {
    'hidden1': np.zeros(n_hidden1),
    'hidden2': np.zeros(n_hidden2),
    'hidden3': np.zeros(n_hidden3),
    'output': np.zeros(n_outputs)
}

# Implement the forward pass
def forward(X, weights, biases):
    hidden1 = np.dot(X, weights['hidden1']) + biases['hidden1']
    hidden1 = np.maximum(hidden1, 0)  # ReLU activation
    hidden2 = np.dot(hidden1, weights['hidden2']) + biases['hidden2']
    hidden2 = np.maximum(hidden2, 0)  # ReLU activation
    hidden3 = np.dot(hidden2, weights['hidden3']) + biases['hidden3']
    hidden3 = np.maximum(hidden3, 0)  # ReLU activation
    output = np.dot(hidden3, weights['output']) + biases['output']
    output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)  # Softmax activation
   
# Implement the backward pass
def backward(X, y, output, weights, biases):
    error = output - y
    d_output = error / len(X)
    d_hidden3 = np.dot(d_output, weights['output'].T)
    d_hidden3[hidden3 <= 0] = 0  # ReLU derivative
    d_hidden2 = np.dot(d_hidden3, weights['hidden3'].T)
    d_hidden2[hidden2 <= 0] = 0  # ReLU derivative
    d_hidden1 = np.dot(d_hidden2, weights['hidden2'].T)
    d_hidden1[hidden1 <= 0] = 0  # ReLU derivative
    gradients = {
        'hidden1': np.dot(X.T, d_hidden1),
        'hidden2': np.dot(hidden1.T, d_hidden2),
        'hidden3': np.dot(hidden2.T, d_hidden3),
        'output': np.dot(hidden3.T, d_output)
    }
    return gradients

# Update the weights and biases
def update(weights, biases, gradients, learning_rate):
    weights['hidden1'] -= learning_rate * gradients['hidden1']
    weights['hidden2'] -= learning_rate * gradients['hidden2']
    weights['hidden3'] -= learning_rate * gradients['hidden3']
    weights['output'] -= learning_rate * gradients['output']
    biases['hidden1'] -= learning_rate * np.mean(gradients['hidden1'], axis=0)
    biases['hidden2'] -= learning_rate * np.mean(gradients['hidden2'], axis=0)
    biases['hidden3'] -= learning_rate * np.mean(gradients['hidden3'], axis=0)
    biases['output'] -= learning_rate * np.mean(gradients['output'], axis=0)

# Train the FNN
def train(X_train, y_train, X_val, y_val, n_epochs, batch_size, learning_rate):
    n_batches = len(X_train) // batch_size
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            # Forward pass
            hidden1, hidden2, hidden3, output = forward(X_batch, weights, biases)
            # Backward pass
            gradients = backward(X_batch, y_batch, output, weights, biases)
            # Update weights and biases
            update(weights, biases, gradients, learning_rate)
        # Compute loss and accuracy on validation set
        _, _, _, val_output = forward(X_val, weights, biases)
        val_loss = np.mean(-np.log(val_output[np.arange(len(y_val)), y_val]))
        val_acc = np.mean(np.argmax(val_output, axis=1) == y_val)
        print(f'Epoch {epoch+1}/{n_epochs}, validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}')

# Evaluate the FNN
def evaluate(X_test, y_test, weights, biases):
    _, _, _, test_output = forward(X_test, weights, biases)
    test_loss = np.mean(-np.log(test_output[np.arange(len(y_test)), y_test]))
    test_acc = np.mean(np.argmax(test_output, axis=1) == y_test)
    print(f'Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}')
