import pandas as pd

# Load the dataset
df = pd.read_csv("🎬 OTT Usage Survey  (Responses) - Form Responses 1.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df_clean = df.drop(columns=["Timestamp", "E-mail"])

# Fill missing feedback
df_clean["Any suggestions or feedback about OTT platforms?"] = df_clean["Any suggestions or feedback about OTT platforms?"].fillna("No feedback")

# Normalize and clean entries
df_clean["Gender"] = df_clean["Gender"].str.strip().str.capitalize()
df_clean["Do you plan to continue using OTT platforms in the future?"] = df_clean["Do you plan to continue using OTT platforms in the future?"].str.strip().str.capitalize()

# Standardize list-like entries
def standardize_list(entry):
    if isinstance(entry, str):
        return ', '.join(sorted([x.strip().capitalize() for x in entry.split(',')]))
    return entry

df_clean["Which OTT platforms do you currently use?"] = df_clean["Which OTT platforms do you currently use?"].apply(standardize_list)
df_clean["What type of content do you watch the most on OTT?"] = df_clean["What type of content do you watch the most on OTT?"].apply(standardize_list)

# Convert to integer
df_clean["How satisfied are you with your preferred OTT platform?"] = df_clean["How satisfied are you with your preferred OTT platform?"].astype(int)

# Save the final cleaned dataset
df_clean.to_csv("OTT_Usage_Survey_Preprocessed.csv", index=False)

print(" Preprocessing complete! File saved as: OTT_Usage_Survey_Preprocessed.csv")
