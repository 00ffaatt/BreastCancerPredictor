import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

## imports the training dataset provided from the link https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset?resource=download
url = "https://raw.githubusercontent.com/00ffaatt/BreastCancerPredictor/main/Breast_cancer_dataset.csv"
df = pd.read_csv(url)

# trimming data
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
# The diagnosis is what we are after as the dependent data (y)
X = df.iloc[:, 1:31]
y = df.iloc[:, 0]

# accuracy score before preprocessing of data: 0.9473684210526315


# PREPROCESSING DATA
# Standard scaling
stdScaler = StandardScaler()
X = stdScaler.fit_transform(X)
# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# *** PCA was performed after standardization of data to avoid biases in components from
# inputs params with different scales ***


# Accuracy score after preprocessing: 0.9649122807017544


# helper method for visualizing confusion matrix
def print_confusion_matrix(cm):
    print("True Positive: " + str(cm[0][0]) + " | False Positive: " + str(cm[0][1]))
    print("False Negative: " + str(cm[1][0]) + " | True Negative: " + str(cm[1][1]))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

print_confusion_matrix(cm)
print("Model accuracy score: " + str(score))

X_reconstructed = pca.inverse_transform(X_pca)
recon_loss = np.mean((X - X_reconstructed) ** 2)
print("PCA reconstruction loss: " + str(recon_loss))
