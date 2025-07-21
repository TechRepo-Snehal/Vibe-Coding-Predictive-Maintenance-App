# 1. SETUP: IMPORT LIBRARIES
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# 2. DATA LOADING
# =============================
# Load the dataset from the CSV file into a pandas DataFrame.
# Ensure the 'ai4i2020.csv' file is in the same directory as this script.
try:
    df = pd.read_csv('ai4i2020.csv')
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'ai4i2020.csv' not found.")
    print("Please download the dataset and place it in the same folder as this script.")
    exit()

# 3. DATA PREPROCESSING
# =============================
# The 'Product ID' is just an identifier and not useful for prediction, so we drop it.
df = df.drop('UDI', axis=1)
df = df.drop('Product ID', axis=1)

# The 'Type' column is categorical ('L', 'M', 'H'). We need to convert it
# into numerical format. We'll use one-hot encoding for this.
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Separate the features (X) from the target variable (y)
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

print("\nData preprocessed. Features are ready for the model.")
print("Feature columns:", X.columns.tolist())

# 4. TRAIN-TEST SPLIT
# =============================
# Split the data into a training set (80%) and a testing set (20%).
# The training set is for teaching the model, and the testing set is for evaluating it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split into training and testing sets.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 5. MODEL TRAINING
# =============================
# We will use a RandomForestClassifier. It's an ensemble of decision trees
# and is very effective for this kind of problem.
# n_estimators=100 means it will build 100 decision trees.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
print("\nTraining the Random Forest model...")
model.fit(X_train, y_train)

print("Model training complete!")

# Show feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 6. EVALUATION
# =============================
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on the Test Set: {accuracy:.4f}")



# 7. PREDICTING PROBABILITY ON NEW DATA (EXAMPLE)
# =============================
# Now, let's generate random new sensor data each time this section runs.
import random
import numpy as np

print("\n--- Simulating a Real-World Prediction with Random Data ---")

# Define ranges for each feature (adjust as needed based on your data)
feature_ranges = {
    'Air temperature [K]': (295, 320),
    'Process temperature [K]': (300, 340),
    'Rotational speed [rpm]': (1200, 1800),
    'Torque [Nm]': (30, 80),
    'Tool wear [min]': (0, 250),
    'TWF': (0, 1),
    'HDF': (0, 1),
    'PWF': (0, 1),
    'OSF': (0, 1),
    'RNF': (0, 1),
    'Type_L': (0, 1),
    'Type_M': (0, 1)
}

# Generate random values for each feature
random_data = {}
for col in X_train.columns:
    if col in feature_ranges:
        low, high = feature_ranges[col]
        # For binary columns, use random.randint; for others, use random.uniform
        if high == 1:
            random_data[col] = [random.randint(low, high)]
        else:
            random_data[col] = [round(random.uniform(low, high), 2)]
    else:
        # If column is not in feature_ranges, fill with 0 or False
        random_data[col] = [0]

new_df = pd.DataFrame(random_data)
new_df = new_df[X_train.columns] # Ensure the order of columns is the same

# Use the trained model to predict the probability
failure_probability = model.predict_proba(new_df)

print(f"\nNew Sensor Data:")
print(new_df)
print(f"\nPredicted Probability of Failure: {failure_probability[0][1]:.2%}")

if failure_probability[0][1] > 0.5:
    print("Prediction: High risk of failure. Maintenance recommended.")
else:
    print("Prediction: Low risk of failure. Machine operating normally.")
