import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print("Number of fraudulent transactions:", transactions['isFraud'].sum())

# Summary statistics on amount column
print(transactions['amount'].describe())

# Create isPayment field
transactions['isPayment'] = transactions['type'].apply(lambda x: 1 if x in ['PAYMENT', 'DEBIT'] else 0)
print(transactions.head())

# Create isMovement field
transactions['isMovement'] = transactions['type'].apply(lambda x: 1 if x in ['CASH_OUT', 'TRANSFER'] else 0)
print(transactions.head())

# Create accountDiff field
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])
print(transactions.head())

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']].values
label = transactions['isFraud'].values
print(features)
print(label)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)

# Normalize the features variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Score the model on the training data
print("Training score:", model.score(X_train, y_train))

# Score the model on the test data
print("Test score:", model.score(X_test, y_test))

# Print the model coefficients
print("Model coefficients:", model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([98765.43, 1.0, 0.0, 21098.765])

# Combine new transactions into a single array
sample_transactions = np.vstack((transaction1, transaction2, transaction3, your_transaction))
print(sample_transactions)

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
fraud_probabilities = model.predict_proba(sample_transactions)
print("Fraud probabilities for new transactions:", fraud_probabilities)
