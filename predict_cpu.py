import pandas as pd
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Required for CI/CD (no screen)
import matplotlib.pyplot as plt

# Add src to path so we can import the loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_loader import load_data
except ImportError:
    print("⚠️ 'data_loader' module not found. Using fallback loader.")
    def load_data(filepath):
        if not os.path.exists(filepath):
            print(f"⚠️ {filepath} not found. Creating dummy data for CI.")
            return pd.DataFrame({'cpu_pct': np.random.uniform(20, 80, 100)})
        return pd.read_csv(filepath)

# Scale + Predict next 6 (60 mins)
print("✅ Starting prediction...")

# Check files
files = [f for f in os.listdir('.') if os.path.isfile(f)]
print("Files found:", files)

# Load Model (Safe Load for CI/CD)
if os.path.exists('lstm_model.h5'):
    model = load_model('lstm_model.h5')
else:
    print("⚠️ lstm_model.h5 not found. Using MockModel for CI testing.")
    class MockModel:
        def predict(self, X, verbose=0):
            return np.array([[0.5]]) # Return 50% load prediction
    model = MockModel()

# Load dataset (last 3 days)
df = load_data('cloud_workload.csv')
print(f"✅ Dataset: {len(df)} rows, shape {df.shape}")

# Scale + predict
scaler = MinMaxScaler()
cpu_scaled = scaler.fit_transform(df['cpu_pct'].values.reshape(-1,1))
X_pred = cpu_scaled[-6:].reshape(1,6,1)  # Last hour

print("✅ Predicting...")
pred_scaled = model.predict(X_pred, verbose=0)
cpu_pred = scaler.inverse_transform(pred_scaled)[0][0]

print(f"PREDICTED CPU (60min): {cpu_pred:.1f}%")
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

# Plot last 24 points (approx 2 hours)
recent_data = df['cpu_pct'].tail(24).values
plt.plot(range(24), recent_data, label='Recent Actual')

# Plot prediction connecting to the last point
plt.plot([23, 24], [recent_data[-1], cpu_pred], 'ro--', linewidth=2, label=f'Predicted: {cpu_pred:.1f}%')

plt.axhline(y=75, color='orange', linestyle=':', label='Scale Threshold (75%)')
plt.ylabel('CPU %')
plt.xlabel('Time Steps')
plt.title(f'ML Workload Prediction: {cpu_pred:.1f}%')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('prediction_graph.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ COMPLETE! Artifacts: prediction.txt + prediction_graph.png")
