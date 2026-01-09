import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("✅ Starting prediction...")

# Check files
files = [f for f in os.listdir('.') if os.path.isfile(f)]
print("Files found:", files)

if not os.path.exists('lstm_model.h5'):
    print("❌ ERROR: lstm_model.h5 missing!")
    exit(1)
if not os.path.exists('cloud_workload.csv'):
    print("❌ ERROR: cloud_workload.csv missing!")
    exit(1)

# Load dataset (last 3 days)
df = pd.read_csv('cloud_workload.csv')
print(f"✅ Dataset: {len(df)} rows, shape {df.shape}")
recent_data = df['cpu_pct'].tail(432).values.reshape(-1, 1)

# Scale + predict
scaler = MinMaxScaler()
recent_scaled = scaler.fit_transform(recent_data)
X_pred = recent_scaled[-6:].reshape(1, 6, 1)

print("✅ Loading model...")
model = load_model('lstm_model.h5')
print("✅ Predicting...")
pred_scaled = model.predict(X_pred, verbose=0)
cpu_pred = scaler.inverse_transform(pred_scaled)[0][0]

print(f"🎯 PREDICTED CPU (60min): {cpu_pred:.1f}%")

# Decision
with open('prediction.txt', 'w') as f:
    f.write(f"{cpu_pred:.1f}")

if cpu_pred > 75:
    print("🚀 AUTO-SCALING TRIGGERED!")
else:
    print("✅ Normal operation")

# Graph
plt.figure(figsize=(10, 4))
plt.plot(recent_data[-24:].flatten(), label='Last 4hrs Actual')
plt.plot([23, 24], [recent_data[-1, 0], cpu_pred], 'ro-', linewidth=3, label=f'Predicted: {cpu_pred:.1f}%')
plt.axhline(75, color='orange', ls='--', label='Scale Threshold (75%)')
plt.ylabel('CPU %')
plt.xlabel('10min Intervals')
plt.legend()
plt.title('ML Autoscaling Prediction')
plt.savefig('prediction_graph.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ COMPLETE! Artifacts: prediction.txt + prediction_graph.png")
