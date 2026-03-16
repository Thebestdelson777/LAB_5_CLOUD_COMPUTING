import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv("data/train_processed.csv")

# Quick look at the data
print("Head of dataset:")
print(df.head())

print("\nDataset description:")
print(df.describe())

# Plot sensor behavior for engine 1
engine_id = 1
engine = df[df.engine_id == engine_id]

plt.figure(figsize=(12,6))
plt.plot(engine["cycle"], engine["sensor_1"], label="Sensor 1")
plt.plot(engine["cycle"], engine["sensor_2"], label="Sensor 2")
plt.title(f"Sensor readings over cycles for Engine {engine_id}")
plt.xlabel("Cycle")
plt.ylabel("Sensor Value")
plt.legend()
plt.show()

# Optional: correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of All Features")
plt.show()
