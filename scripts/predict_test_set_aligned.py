import pandas as pd
import joblib

# Load test features
X_test = pd.read_csv("features/tsfresh_features_test_selected.csv")
print("Original test shape:", X_test.shape)

# Rename columns to match model training (0,1,2,...)
X_test.columns = [str(i) for i in range(X_test.shape[1])]
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
    "features/test_set_predictions_aligned.csv", index=False
)
print("Predictions saved to features/test_set_predictions_aligned.csv")
