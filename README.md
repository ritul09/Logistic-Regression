# logistic_regression
Performing logistic regression on the dataset using numpy only (no sklearn).
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('data.csv')

# Create the features and target variables
features = data[['feature_1', 'feature_2']]
target = data['target']

# Create the logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(features, target)

# Make predictions
predictions = model.predict(features)

# Evaluate the model
print(model.score(features, target))
