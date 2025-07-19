import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load
df = pd.read_csv("OTT_Survey_50_Responses.csv")

# 2. Basic cleaning
df = df.drop_duplicates()
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# 3. One‑hot encode categoricals
categorical_cols = [
    "Gender", "Preferred Platform", "Favourite Genre",
    "Subscription Type", "Affects Academics (Yes/No)"
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# 4. Scale numerics
scaler = MinMaxScaler()
df[["Age", "Daily Usage (hrs)"]] = scaler.fit_transform(
    df[["Age", "Daily Usage (hrs)"]]
)

# 5. Save
df.to_csv("OTT_Survey_50_Responses_Preprocessed.csv", index=False)
print("Preprocessing complete!")
