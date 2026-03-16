import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ----------------------------
# Load features
# ----------------------------
X = pd.read_csv("features/tsfresh_features_selected.csv")

# ----------------------------
# Load target RUL and aggregate to match features
# ----------------------------
y_full = pd.read_csv("data/train_processed.csv")

# Aggregate RUL per engine (assuming one feature row per engine)
y = y_full.groupby('engine_id')['RUL'].max()
# Match order to X if engine_id column exists in features
if 'engine_id' in X.columns:
    y = y.loc[X['engine_id']].values
else:
    y = y.values  # if no engine_id, assume same order

# ----------------------------
# Split into train and validation sets
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Train Random Forest
# ----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ----------------------------
# Predict and evaluate
# ----------------------------
y_pred = rf.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("Validation MSE:", mse)

# ----------------------------
# Save the trained model
# ----------------------------
joblib.dump(rf, "models/random_forest_rul.pkl")
print("Model saved to models/random_forest_rul.pkl")
