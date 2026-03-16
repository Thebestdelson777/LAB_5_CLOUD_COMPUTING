import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

# Load processed data
df = pd.read_csv("data/train_processed.csv")

# Optional: reduce number of engines for faster testing
# df = df[df.engine_id <= 30]

# Use EfficientFCParameters dict
fc_params = EfficientFCParameters()

# Extract features
features = extract_features(
    df,
    column_id="engine_id",
    column_sort="cycle",
    default_fc_parameters=fc_params
)

# Save features
features.to_csv("features/tsfresh_features.csv")
print("Feature extraction complete")
print("Features shape:", features.shape)
