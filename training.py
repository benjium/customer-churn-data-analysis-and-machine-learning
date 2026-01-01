#%%
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

#%%
# Load data
df = pd.read_excel('customer chun dataset.xlsx')
print(f"Initial shape: {df.shape}")
df.head()

# %%
# Drop missing values
df = df[~df['TotalCharges'].isna()].copy()
print(f"Shape after dropping missing TotalCharges: {df.shape}")

# %%
# Convert numeric columns
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# Define categorical columns
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'
]

# Convert to category type
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

#%%
# Drop customerID
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

#%%
# Ensure Churn is categorical and create numeric binary column
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].astype('category')
    df['churn_flag'] = (df['Churn'] == 'Yes').astype(int)  # Explicit mapping

print(f"Churn distribution:\n{df['churn_flag'].value_counts()}")

# %%
# Define features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

# Verify all categorical columns exist in the dataframe
categorical_features = [col for col in categorical_cols if col in df.columns]

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# %%
# Prepare modeling DataFrame - keep all features
model_features = numeric_features + categorical_features
model_df = df[model_features + ['churn_flag']].dropna().copy()

print(f"\nFinal modeling shape: {model_df.shape}")
print(f"Missing values:\n{model_df.isnull().sum()}")

# %%
if model_df.shape[0] < 50:
    print("Not enough data for modeling.")
else:
    # Split features and target
    X = model_df[model_features]
    y = model_df['churn_flag'].values

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # %%
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # %%
    # Build preprocessor pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # %%
    # Build model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete!")

    # %%
    # Make predictions and evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"{'='*50}")

    # %%
    # Save the model
    joblib.dump(pipeline, 'model.pkl')
    print('\n✅ Model saved successfully as model.pkl')