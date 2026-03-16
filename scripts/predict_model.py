import pandas as pd
import joblib

# --------------------------
# Load test data
# --------------------------
test_df = pd.read_csv("data/test_FD001.txt", delim_whitespace=True, header=None)
# Add column names matching train_processed.csv
columns = ['engine_id', 'cycle'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]
test_df.columns = columns

# --------------------------
# Preprocess test data
# --------------------------
# Example: compute RUL placeholder if needed
# For prediction we usually just need features like in train
# If you did any normalization/scaling in training, apply same here
# For simplicity, we'll assume train preprocessing is compatible
# (Otherwise, import preprocess function and apply)

# --------------------------
# Load trained model
# --------------------------
model = joblib.load("models/random_forest_rul.pkl")
print("Model loaded successfully.")

# --------------------------
# Load selected features
# --------------------------
# Keep same columns as used in training
selected_features = pd.read_csv("features/tsfresh_features_selected.csv").columns
X_test = test_df[selected_features]

# --------------------------
# Predict RUL
# --------------------------
y_pred = model.predict(X_test)
print("Predictions complete.")

# --------------------------
# Save predictions
# --------------------------
output_df = pd.DataFrame({
    'engine_id': test_df['engine_id'],
    'cycle': test_df['cycle'],
    'predicted_RUL': y_pred
})
output_df.to_csv("predictions/test_FD001_predictions.csv", index=False)
print("Predictions saved to predictions/test_FD001_predictions.csv")
