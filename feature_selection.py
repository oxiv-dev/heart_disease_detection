import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("data/processed.csv")
    print(data.head())
    print(data.shape)
    return data

def l2_regularization_lr(alpha=1.0):

    ds = load_data()
    X = ds.iloc[:, :37].astype(float)
    y = ds.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=3)
    sel = SelectFromModel(LogisticRegression(C=10.0, solver='liblinear'))
    sel.fit(X_train, y_train)
    
    return sel.get_feature_names_out()

def mutual_info(X, y):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    # Compute entropy of the target variable
    p_y = np.bincount(y) / n_samples
    H_y = -np.sum(p_y * np.log2(p_y + 1e-10))

    for i in range(n_features):
        # Compute entropy of the i-th feature
        feature = X.iloc[:, i]
        bins = np.histogram_bin_edges(feature, bins='fd')
        hist_2d, _, _ = np.histogram2d(feature, y, bins=[bins, [0,1]])
        p_xy = hist_2d / n_samples
        p_x = np.sum(p_xy, axis=1)
        p_y_given_x = p_xy / (p_x[:, np.newaxis] + 1e-10)
        H_y_given_x = -np.sum(p_y_given_x * np.log2(p_y_given_x + 1e-10), axis=1)
        mi_scores[i] = np.sum(p_x * H_y_given_x) - H_y

    return mi_scores


def mutual_info_selection():
    ds = load_data()
    X = ds.iloc[:, :37].astype(float)
    y = ds.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    mi_scores = mutual_info(X, y)
    feature_indices = np.argsort(mi_scores)[::-1][:18]
    labels = [X.columns[i] for i in feature_indices]
    
    return labels
