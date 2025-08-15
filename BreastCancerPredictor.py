import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


df = pd.read_csv("https://github.com/00ffaatt/BreastCancerPredictor/blob/main/Breast_cancer_dataset.csv")


print(df)
