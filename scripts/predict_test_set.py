import pandas as pd
import joblib

# Load test features
X_test = pd.read_csv("features/tsfresh_features_test.csv")

# Load selected features (same as used in training)
selected_features = pd.read_csv("features/selected_feature_list.csv", header=None).squeeze().tolist()
selected_features = [str(f).strip() for f in selected_features if str(f).strip() != ""]
print(f"Total selected features: {len(selected_features)}")

# Ensure test set has all features the model expects
for f in selected_features:
    if f not in X_test.columns:
        X_test[f] = 0  # Fill missing columns with 0

# Keep features in the exact order
X_test_selected = X_test[selected_features]

print("Shape of test features after alignment:", X_test_selected.shape)

# Load model
model = joblib.load("models/random_forest_rul.pkl")
print("Model loaded successfully.")

# Predict
predictions = model.predict(X_test_selected)
print("Predictions shape:", predictions.shape)
print("First 10 predictions:", predictions[:10])

# Save predictions
pd.DataFrame(predictions, columns=["RUL_prediction"]).to_csv(
    "features/test_set_predictions.csv", index=False
)
print("Predictions saved to features/test_set_predictions.csv")
