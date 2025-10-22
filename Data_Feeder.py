import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Data_Generator import NiftiDataGenerator

# Load phenotypic data
df = pd.read_csv('./Phenotypic_file.csv')

# Define features to use
features = ['AGE_AT_SCAN', 'SEX', 'HANDEDNESS_CATEGORY', 'FIQ', 'VIQ', 'PIQ']
file_names = []
labels = []
phenotypic_data = []

# Process files and corresponding phenotypic data
processed_files_path = "./New_Processed/New_Processed"
for file_name in os.listdir(processed_files_path):
    sub_id = int(file_name.split(".")[0])
    record = df[df['SUB_ID'] == sub_id]

    if not record.empty:
        dx_group = record['DX_GROUP'].iloc[0]
        if dx_group in [1, 2]:
            file_names.append(os.path.join(processed_files_path, file_name))
            labels.append(1 if dx_group == 1 else 0)

            # Extract and handle missing phenotypic data (impute with mean)
            pheno_features = record[features].iloc[0]
            phenotypic_data.append(pheno_features)

# Create a DataFrame for phenotypic data
phenotypic_df = pd.DataFrame(phenotypic_data)

# Impute missing values with the mean
for col in phenotypic_df.columns:
    if phenotypic_df[col].isnull().any():
        phenotypic_df[col].fillna(phenotypic_df[col].mean(), inplace=True)

# Scale phenotypic data
scaler = StandardScaler()
phenotypic_scaled = scaler.fit_transform(phenotypic_df)
phenotypic_scaled_df = pd.DataFrame(phenotypic_scaled, columns=features)

# Split data into training and validation sets
X_train_files, X_val_files, y_train, y_val, pheno_train, pheno_val = train_test_split(
    file_names, labels, phenotypic_scaled_df, test_size=0.2, random_state=42
)

# Create data generators
train_generator = NiftiDataGenerator(X_train_files, y_train, pheno_train, augment=True)
val_generator = NiftiDataGenerator(X_val_files, y_val, pheno_val, augment=False)

print(f"Training samples: {len(X_train_files)}")
print(f"Validation samples: {len(X_val_files)}")
