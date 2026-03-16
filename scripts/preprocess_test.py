import pandas as pd

# Load raw test data
test_df = pd.read_csv("data/test_FD001.txt", delim_whitespace=True, header=None)

# Add column names (same as training)
columns = ['engine_id', 'cycle'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]
test_df.columns = columns

# Save processed test data
test_df.to_csv("data/test_processed.csv", index=False)
print("Test data preprocessed: test_processed.csv created")
