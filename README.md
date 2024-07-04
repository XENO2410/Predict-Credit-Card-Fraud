# Predict Credit Card Fraud

## Overview
Credit card fraud is one of the leading causes of identity theft around the world. In 2018 alone, over $24 billion were stolen through fraudulent credit card transactions. Financial institutions employ a wide variety of different techniques to prevent fraud, one of the most common being Logistic Regression.

In this project, you are a Data Scientist working for a credit card company. You have access to a dataset (based on a synthetic financial dataset) that represents a typical set of credit card transactions. `transactions.csv` is the original dataset containing 200k transactions

### Load the Data
1. The file `transactions.csv` contains data on 1000 simulated credit card transactions. Begin by loading the data into a pandas DataFrame named `transactions`. Take a peek at the dataset using `.head()` and use `.info()` to examine how many rows there are and what datatypes they are. How many transactions are fraudulent? Print your answer.

### Clean the Data
2. Calculate summary statistics for the `amount` column. What does the distribution look like?
3. Create a new column called `isPayment` that assigns a 1 when `type` is “PAYMENT” or “DEBIT”, and a 0 otherwise.
4. Create a column called `isMovement`, which will capture if money moved out of the origin account. This column will have a value of 1 when `type` is either “CASH_OUT” or “TRANSFER”, and a 0 otherwise.
5. Create a column called `accountDiff` with the absolute difference of the `oldbalanceOrg` and `oldbalanceDest` columns.

### Select and Split the Data
6. Define your features and label columns. Create a variable called `features` consisting of `amount`, `isPayment`, `isMovement`, `accountDiff`, and a variable called `label` with the column `isFraud`.
7. Split the data into training and test sets using sklearn's `train_test_split()` method with a test_size value of 0.3.

### Normalize the Data
8. Scale your feature data using sklearn's `StandardScaler`. Fit and transform the scaler on the training features, and transform the test features.

### Create and Evaluate the Model
9. Create a `LogisticRegression` model with sklearn and fit it on the training data.
10. Score the model on the training data and print the training score.
11. Score the model on the test data and print the test score. How did your model perform?
12. Print the coefficients for the model to see how important each feature was for prediction. Which feature was most important? Least important?

### Predict With the Model
13. Use the model to process new transactions. Three numpy arrays (`transaction1`, `transaction2`, `transaction3`) with information on new sample transactions are pre-loaded in the workspace. Create a fourth array, `your_transaction`, with your own values.
14. Combine the new transactions and `your_transaction` into a single numpy array called `sample_transactions`.
15. Scale `sample_transactions` using the `StandardScaler`.
16. Use the model's `predict()` method on `sample_transactions` to determine which transactions are fraudulent. Print the results.
17. Call the model's `predict_proba()` method on `sample_transactions` and print the result to see the probabilities that led to these predictions.


## Further Exploration
- Check how many fraudulent transactions are there in the complete dataset. What percentage of the total number of transactions is this? Compare this percentage to that in the modified dataset.

