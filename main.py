from feature_selection import load_data, l2_regularization_lr, mutual_info_selection
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nn4 import FNN

df = load_data()
X = df[mutual_info_selection()]
y = df.iloc[:, -1]

print(X.head())
print(y)
# print(df['DIAGNOSIS'].value_counts())
# print(f"L2 ---------------------\n#{l2_regularization_lr()}")

# print(f"MI ---------------------\n#{mutual_info_selection()}")

#balancing!!!!!!!!!!!!!!!!!!!!!

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

from sklearn.utils.class_weight import compute_class_weight

# Assuming y_train is your training labels array
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# Create dictionary of class weights
class_weights_dict = dict(enumerate(class_weights))

model = FNN(num_inputs=18, num_outputs=31, hidden_layer_sizes=[128, 64], threshold=0.5, activation="sigmoid")
model.fit(X, y, num_epochs=50, learning_rate=0.5, batch_size=32)

# Pass the class weights dictionary to the fit method of your neural network model

# y_test_arg=y_test.reshape(-1,1)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
enc = LabelEncoder()
y_test = enc.fit_transform(y_test)
print(y_pred)
print(y_test)

print('Confusion Matrix')
print(multilabel_confusion_matrix(y_test, y_pred))

# f, axes = plt.subplots(5, 7, figsize=(25, 15))
# axes = axes.ravel()
# for i in range(31):
#     disp = ConfusionMatrixDisplay(confusion_matrix(y_test[:, i],
#                                                    Y_pred[:, i]),
#                                   display_labels=[0, i])
#     disp.plot(ax=axes[i], values_format='.4g')
#     disp.ax_.set_title(f'class {i}')
#     if i<10:
#         disp.ax_.set_xlabel('')
#     if i%5!=0:
#         disp.ax_.set_ylabel('')
#     disp.im_.colorbar.remove()

# plt.subplots_adjust(wspace=0.10, hspace=0.1)
# f.colorbar(disp.im_, ax=axes)
# plt.show()

cm = confusion_matrix(y_test, y_pred)

# Get the class labels
classes = np.unique(np.concatenate((y_test, y_pred)))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



# y_pred = model.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)



