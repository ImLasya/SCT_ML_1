# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # Define features and target FIRST
# features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
# target = 'SalePrice'

# # Load datasets
# train_df = pd.read_csv('./train.csv')
# test_df = pd.read_csv('./test.csv')

# # Extract inputs and target
# X = train_df[features]
# y = train_df[target]
# X_test = test_df[features]

# # Check for missing values
# print("Missing values in training features:\n", X.isnull().sum())
# print("Missing values in test features:\n", X_test.isnull().sum())
# print("Missing values in target:\n", y.isnull().sum())

# # Fill missing values with mean
# imputer = SimpleImputer(strategy='mean')
# X = pd.DataFrame(imputer.fit_transform(X), columns=features)
# X_test = pd.DataFrame(imputer.transform(X_test), columns=features)

# # Train-validation split
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Validation predictions
# val_preds = model.predict(X_val)

# # Evaluation metrics
# r2 = r2_score(y_val, val_preds)
# rmse = np.sqrt(mean_squared_error(y_val, val_preds))

# print("\n📊 Evaluation Metrics:")
# print(f"R² Score: {r2:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# # Predict on test dataset
# test_preds = model.predict(X_test)

# # Submission file
# submission = pd.DataFrame({
#     'Id': test_df['Id'],
#     'SalePrice': test_preds
# })
# submission.to_csv('submission.csv', index=False)
# print("\n✅ submission.csv created.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Setup
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Define features and target
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Load data
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Extract data
X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# Visualize feature vs target
for feature in features:
    sns.scatterplot(data=train_df, x=feature, y=target)
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.tight_layout()
    plt.savefig(f"{feature}_vs_SalePrice.png")
    plt.clf()  # Clear figure

# Check missing values
print("Missing values in training features:\n", X.isnull().sum())
print("Missing values in test features:\n", X_test.isnull().sum())
print("Missing values in target:\n", y.isnull().sum())

# Fill missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)
X_test = pd.DataFrame(imputer.transform(X_test), columns=features)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
val_preds = model.predict(X_val)

# Evaluation
r2 = r2_score(y_val, val_preds)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("\n📊 Evaluation Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 🔍 Residual Plot
residuals = y_val - val_preds
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("residuals_distribution.png")
plt.clf()

# 🔍 Actual vs Predicted Plot
sns.scatterplot(x=y_val, y=val_preds, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted Sale Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.clf()

# Predict on test data
test_preds = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_preds
})
submission.to_csv('submission.csv', index=False)

print("✅ submission.csv created.")
print("📸 Visualizations saved as PNG files.")

