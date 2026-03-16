import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np

# Load extracted features from training
X = pd.read_csv("features/tsfresh_features.csv", index_col=0)

# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with column mean
X.fillna(X.mean(), inplace=True)

print("Cleaned shape:", X.shape)

# Apply variance threshold to remove near-constant features
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Keep the selected column names
selected_features = X.columns[selector.get_support()]

# Convert back to DataFrame for saving
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
X_selected_df.to_csv("features/tsfresh_features_selected.csv")

# Save list of selected feature names for test set
pd.Series(selected_features).to_csv("features/selected_feature_list.csv", index=False)

print("After variance filter:", X_selected_df.shape)
print("Selected features saved to tsfresh_features_selected.csv")
print("List of feature names saved to selected_feature_list.csv")
