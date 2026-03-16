# scripts/feature_selection_test.py

import pandas as pd

# 1️⃣ Load test features
X_test = pd.read_csv("features/tsfresh_features_test.csv")
print("Original test shape:", X_test.shape)
print("First 5 columns of test set:", X_test.columns[:5].tolist())

# 2️⃣ Load selected features from training
# We use header=None and squeeze to get a flat list 
selected_features = pd.read_csv("features/selected_feature_list.csv", header=None).squeeze().tolist()

# 2a️⃣ Remove any numeric index or empty strings that might appear
selected_features = [str(f).strip() for f in selected_features if str(f).strip() != ""]

print(f"Total features in selected list: {len(selected_features)}")
print("First 10 features from selected list:", selected_features[:10])

# 3️⃣ Keep only features that exist in X_test
existing_features = [f for f in selected_features if f in X_test.columns]
missing_features = [f for f in selected_features if f not in X_test.columns]

print(f"Number of features found in test set: {len(existing_features)}")
print(f"Number of features missing from test set: {len(missing_features)}")
if missing_features:
    print("Some missing features:", missing_features[:10], "...")

# 4️⃣ Subset test set
X_test_selected = X_test[existing_features]
print("Shape after selecting features:", X_test_selected.shape)

# 5️⃣ Save the cleaned test features
X_test_selected.to_csv("features/tsfresh_features_test_selected.csv", index=False)
print("Selected test features saved to features/tsfresh_features_test_selected.csv")
