import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the ELU activation function
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Define the loglog activation function
def loglog(x):
    return np.log(-np.log(1 - x))

# Define the neural network architecture
input_size = 18
hidden_size_1 = 64
hidden_size_2 = 32
output_size = 8

# Define the weights for each layer
W1 = np.random.randn(input_size, hidden_size_1)
W2 = np.random.randn(hidden_size_1, hidden_size_2)
W3 = np.random.randn(hidden_size_2, output_size)

# Define the bias for each layer
b1 = np.zeros((1, hidden_size_1))
b2 = np.zeros((1, hidden_size_2))
b3 = np.zeros((1, output_size))

# Define the input data
X = np.random.rand(1, input_size)

# Feedforward computation
z1 = np.dot(X, W1) + b1
a1 = elu(z1)
z2 = np.dot(a1, W2) + b2
a2 = elu(z2)
z3 = np.dot(a2, W3) + b3
a3 = sigmoid(z3)

# Convert the predictions to class labels
y_pred_labels = np.argmax(a3, axis=1)

# Define the threshold for ambiguity
threshold = 0.5

# Define the output classes
classes = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']

# Create a list to hold the final predictions
final_predictions = []

# Iterate over the predicted labels
for label in y_pred_labels:
    # If the predicted probability is below the threshold, mark the prediction as undefined
    if np.max(a3[label]) < threshold:
        final_predictions.append('undefined')
    # Otherwise, output the predicted class
    else:
        final_predictions.append(classes[label])

# Print the final predictions
print(final_predictions)
