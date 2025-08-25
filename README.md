# Python program using sklearn to predict Breast cancer malignancy using biopsy cell features

Training Logistic Regression model using data from https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset?resource=download

# Steps Taken

## 1. Data Preprocessing

- removed data columns such as ID and the last column with all NaN values
- replaced malignancy column values with binary (1 = Malignant, 2 = Benign)
- Standardized data using sklearn's StandardScaler
- performed PCA with 2 components on the independent data (X)
- Split the data into training and testing groups using a standard seed.

## 2. Training the model

- a Logistic Regression model was trained based on the preprocessed training data.

## 3. Evaluating the model

- Used sklearn's accuracy_score to evaluate the effectiveness of the model before (~0.947)
  and after (~0.965) standardization+PCA analysis
- calculated the PCA reconstruction loss using the inverse_transform function to be ~0.368

## Further exploration

(Some ideas for increasing accuracy)

- Play around with different thresholds for the logistic regression output
- Use different numbers of principle components during PCA: more components for higher accuracy at the cost of computing power
- Trying different types of regressions (e.g. decision tree, regressions with penalty terms)
