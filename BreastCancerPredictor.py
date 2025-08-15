import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


url = "https://raw.githubusercontent.com/00ffaatt/BreastCancerPredictor/main/Breast_cancer_dataset.csv"

df = pd.read_csv(url)




