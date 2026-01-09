import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
import os

# Add src to path so we can import the loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_data

# Load your trained model + dataset
model = load_model('lstm_model.h5')
df = pd.read_csv('cloud_workload.csv')
df = load_data('cloud_workload.csv')
cpu_data = df['cpu_pct'].values[-432:]  # Last 3 days

# Scale + Predict next 6 (60 mins)
scaler = MinMaxScaler()
cpu_scaled = scaler.fit_transform(cpu_data.reshape(-1,1))
X_pred = cpu_scaled[-6:].reshape(1,6,1)  # Last hour
pred_scaled = model.predict(X_pred)[0][0]

# Inverse transform
cpu_pred = scaler.inverse_transform([[pred_scaled]])[0][0]

print(f"PREDICTED CPU (60min): {cpu_pred:.1f}%")
with open('prediction.txt', 'w') as f:
    f.write(f"{cpu_pred:.1f}")

# Graph
plt.figure(figsize=(10,4))
plt.plot(cpu_data[-24:], label='Last 4hrs Actual')
future = np.append(cpu_data[-6:], cpu_pred)
plt.plot(range(22,28), future[-6:], 'r--', label='Predicted')
plt.axhline(y=75, color='orange', ls=':', label='Scale threshold')
plt.legend()
plt.title(f'CPU Prediction: {cpu_pred:.1f}%')
plt.savefig('prediction_graph.png')
plt.close()

print("✅ Pipeline complete!")
