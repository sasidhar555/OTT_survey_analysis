import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load preprocessed dataset
df = pd.read_csv("OTT_Survey_50_Responses_Preprocessed.csv")

# Step 2: Split features and target
X = df.drop(columns=["Affects Academics (Yes/No)_No", "Affects Academics (Yes/No)_Yes"])
y = df["Affects Academics (Yes/No)_Yes"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Evaluation
print(">>> Random Forest Classifier Results <<<")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
