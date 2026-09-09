import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("onlinefraud.csv")
print(df.head())

fig = px.pie(df, names='isFraud', title='Distribution of Fraudulent Transactions')
fig.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


sample_features = np.array([[1, 9839.64, 170136.0, 160296.36, 0.0, 0.0]])
prediction = model.predict(sample_features)
print(f"Fraud Prediction (0 = No, 1 = Yes): {prediction[0]}")
