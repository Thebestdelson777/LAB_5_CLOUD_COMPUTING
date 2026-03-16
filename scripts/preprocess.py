import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)  # remove empty columns
    return df

# Load the training file
train = load_data("data/train_FD001.txt")

# Define column names
cols = ['engine_id','cycle'] + \
       [f'op_setting_{i}' for i in range(1,4)] + \
       [f'sensor_{i}' for i in range(1,22)]

train.columns = cols

# Compute max cycle per engine
max_cycle = train.groupby('engine_id')['cycle'].max()

# Compute RUL
train['RUL'] = train.apply(lambda row: max_cycle[row.engine_id] - row.cycle, axis=1)

# Save processed file
train.to_csv("data/train_processed.csv", index=False)

print("Preprocessing finished: train_processed.csv created")
