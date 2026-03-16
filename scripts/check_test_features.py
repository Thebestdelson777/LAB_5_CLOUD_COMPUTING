import pandas as pd

# Load the test features you aligned
X_test_selected = pd.read_csv("features/tsfresh_features_test_selected.csv")

print("Number of features in test set:", X_test_selected.shape[1])
print("First 20 columns:", X_test_selected.columns[:20].tolist())
