import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("✅ Starting prediction...")

# Verify files exist
print("Files:", [f for f in os.listdir('.') if os.path.exists(f)])
if not os.path.exists('lstm_model.h5'):
    raise FileNotFoundError("❌ lstm_model.h5 missing!")
if not os.path.exists('cloud_workload.csv'):
    raise FileNotFoundError("❌ cloud_workload.csv missing!")

# Load data (last 3 days = 432 rows)
df = pd.read_csv('cloud_workload.csv')
print(f"✅ Dataset loaded: {len(df)} rows")
recent_data = df['cpu_pct'].tail(432).values.reshape(-1, 1)  # Last 3 days

# Scale & predict
scaler = MinMaxScaler()
recent_scaled = scaler.fit_transform(recent_data)
X_pred = recent_scaled[-6:].reshape(1, 6, 1)  # Last 60 mins

print("✅ Model loading...")
model = load_model('lstm_model.h5')
print("✅ Predicting...")
pred_scaled = model.predict(X_pred, verbose=0)[0][0]
cpu_pred = scaler.inverse_transform([[pred_scaled]])[0][0]

print(f"🎯 PREDICTED CPU (60min): {cpu_pred:.1f}%")

# Save
with open('prediction.txt', 'w') as f:
    f.write(f"{cpu_pred:.1f}")

# Quick graph
plt.figure(figsize=(10,3))
plt.plot(recent_data[-24:].flatten(), label='Last 4hrs')
plt.plot([23, 24], [recent_data[-1][0], cpu_pred], 'r--o', label=f'Predicted {cpu_pred:.1f}%')
plt.axhline(75, color='orange', ls=':', label='Scale @75%')
plt.legend()
plt.title('Synthetic Workload Prediction')
plt.savefig('prediction_graph.png', dpi=100, bbox_inches='tight')
plt.close()

print("✅ SUCCESS! Check prediction.txt & prediction_graph.png")
