import joblib

# Load the trained model
model = joblib.load("models/random_forest_rul.pkl")

# Print the features the model expects
print("Number of features the model expects:", len(model.feature_names_in_))
print("First 20 feature names:", model.feature_names_in_[:20])
