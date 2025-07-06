import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the preprocessed dataset
df = pd.read_csv("OTT_Usage_Survey_Preprocessed.csv")

# Set target and features
X = df.drop(columns=["Do you plan to continue using OTT platforms in the future?"])
y = df["Do you plan to continue using OTT platforms in the future?"]

# Encode target variable (Yes/No/Not sure)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Define column transformer to encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Leave numeric columns as-is
)

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train and evaluate both models
for name, model in models.items():
    print(f"\n Model: {name}")
    
    # Create pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Train model
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Evaluation
    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(" Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
