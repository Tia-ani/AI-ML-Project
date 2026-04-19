import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop CustomerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric (it has blanks)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values with 0
df.fillna(0, inplace=True)

# Convert Yes/No to 1/0
df.replace({"Yes": 1, "No": 0}, inplace=True)

# Convert Male/Female
df.replace({"Male": 1, "Female": 0}, inplace=True)

# Convert other categorical columns using one-hot encoding
df = pd.get_dummies(df)

# Save cleaned file
df.to_csv("cleaned_data.csv", index=False)

print("Data cleaned and saved successfully!")
