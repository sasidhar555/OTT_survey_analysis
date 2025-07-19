import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load preprocessed CSV
df = pd.read_csv("OTT_Survey_50_Responses_Preprocessed.csv")

# Split features and target
X = df.drop(columns=["Affects Academics (Yes/No)_No", "Affects Academics (Yes/No)_Yes"])
y = df["Affects Academics (Yes/No)_Yes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(">>> Logistic Regression Results <<<")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
