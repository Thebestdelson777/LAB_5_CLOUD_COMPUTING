import pandas as pd
import joblib

# Load selected test features
X_test = pd.read_csv("features/tsfresh_features_test_selected.csv")
print("Original test shape:", X_test.shape)

# Model expects 9487 features
n_model_features = 9487

# Add missing columns if test set has fewer columns
if X_test.shape[1] < n_model_features:
    for i in range(X_test.shape[1], n_model_features):
        X_test[str(i)] = 0
    print(f"Added {n_model_features - X_test.shape[1]} missing columns to test set.")

# Rename all columns to match model training
X_test.columns = [str(i) for i in range(n_model_features)]
print("Columns renamed to match model.")

# Load model
model = joblib.load("models/random_forest_rul.pkl")
print("Model loaded successfully.")

# Predict
predictions = model.predict(X_test)
print("Predictions shape:", predictions.shape)
print("First 10 predictions:", predictions[:10])

# Save predictions
pd.DataFrame(predictions, columns=["RUL_prediction"]).to_csv(
    "features/test_set_predictions_final.csv", index=False
)
print("Predictions saved to features/test_set_predictions_final.csv")
