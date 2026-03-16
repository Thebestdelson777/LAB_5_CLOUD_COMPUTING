import pandas as pd
from tsfresh import extract_features

# Load processed test data
test_df = pd.read_csv("data/test_processed.csv")

# Extract features
features = extract_features(
    test_df,
    column_id="engine_id",
    column_sort="cycle",
    disable_progressbar=False
)

# Save extracted features
features.to_csv("features/tsfresh_features_test.csv")
print("Test features extracted: tsfresh_features_test.csv")
