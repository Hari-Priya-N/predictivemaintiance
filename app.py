import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(page_title="Industrial AI: LSTM & RUL Sentinel", layout="wide")

# ---------------------------------
# 1. Model Architecture (LSTM Autoencoder)
# ---------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        
        # Encoder: Compresses sequence into a latent representation
        self.encoder = nn.LSTM(n_features, embedding_dim, batch_first=True)
        # Decoder: Attempts to reconstruct the original sequence
        self.decoder = nn.LSTM(embedding_dim, n_features, batch_first=True)
        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        x = hidden.repeat(self.seq_len, 1, 1).transpose(0, 1)
        x, _ = self.decoder(x)
        return self.output_layer(x)

# ---------------------------------
# 2. Core Logic: Data & Training
# ---------------------------------
def get_sensor_reading(t, failure_mode=False):
    """Simulates real-time sensor streams with degradation logic."""
    base_vib = 0.6 + 0.08 * np.sin(t / 5)
    base_temp = 55 + 0.15 * np.cos(t / 10)
    
    if failure_mode and t > 80:
        # Exponential degradation curve
        degradation = np.exp((t - 80) / 25) * 0.04
        v = base_vib + degradation + np.random.normal(0, 0.03)
        temp = base_temp + (degradation * 12) + np.random.normal(0, 0.15)
        return v, temp
    
    return base_vib + np.random.normal(0, 0.02), base_temp + np.random.normal(0, 0.1)

@st.cache_resource
def train_model():
    """Trains the LSTM on a healthy baseline to establish 'Normal' behavior."""
    seq_len, n_features = 12, 2
    model = LSTMAutoencoder(seq_len, n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Baseline simulation
    normal_samples = np.array([get_sensor_reading(i) for i in range(400)])
    scaler = StandardScaler().fit(normal_samples)
    scaled_data = scaler.transform(normal_samples)
    
    sequences = [scaled_data[i:i+seq_len] for i in range(len(scaled_data)-seq_len)]
    seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    
    # Training Loop
    for _ in range(80):
        optimizer.zero_grad()
        loss = criterion(model(seq_tensor), seq_tensor)
        loss.backward(); optimizer.step()
        
    return model, scaler, seq_len

def estimate_rul(error_history, threshold=2.0):
    """Calculates RUL based on the velocity of error increase."""
    if len(error_history) < 15: return None
    
    # Linear fit on the last 15 error points
    y = np.array(error_history[-15:])
    x = np.arange(15)
    slope, intercept = np.polyfit(x, y, 1)
    
    if slope <= 0.001: return "Stable" # No significant degradation trend
    
    # RUL = (Threshold - Current_Value) / Rate_of_Change
    current_error = error_history[-1]
    remaining_capacity = threshold - current_error
    rul_seconds = max(0, remaining_capacity / slope)
    return int(rul_seconds)

# ---------------------------------
# 3. Streamlit Dashboard
# ---------------------------------
st.title("🏭 AI Maintenance Sentinel: LSTM + RUL Engine")
st.markdown("""
Monitor machine health in real-time. This system uses an **LSTM Autoencoder** to learn the 
'signature' of a healthy machine. Deviations generate an anomaly score which is then used to 
predict the **Remaining Useful Life (RUL)**.
""")

model, scaler, seq_len = train_model()
criterion = nn.MSELoss(reduction='none')

# Sidebar
st.sidebar.header("🎛️ Control Center")
start_mon = st.sidebar.button("▶️ Start Live Stream", use_container_width=True)
fail_sim = st.sidebar.checkbox("Trigger Mechanical Wear (Simulate Failure)")
CRITICAL_THRESHOLD = 2.2

# Placeholder elements
kpi_cols = st.columns(3)
chart_holder = st.empty()
log_expander = st.sidebar.expander("📝 Maintenance Log", expanded=True)

if start_mon:
    raw_data_stream = []
    error_stream = []
    logs = []
    
    for t in range(1, 1000):
        # 1. Capture Data
        v, temp = get_sensor_reading(t, fail_sim)
        raw_data_stream.append([v, temp])
        
        if len(raw_data_stream) > seq_len:
            # 2. AI Inference
            recent_data = np.array(raw_data_stream[-seq_len:])
            scaled_seq = scaler.transform(recent_data)
            input_t = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                reconstruction = model(input_t)
                loss = torch.mean(criterion(reconstruction, input_t)).item()
                error_stream.append(loss)
            
            # 3. RUL Logic
            rul = estimate_rul(error_stream, CRITICAL_THRESHOLD)
            
            # 4. Metrics Update
            with kpi_cols[0]:
                st.metric("Vibration (g)", f"{v:.3f}")
            with kpi_cols[1]:
                delta_val = f"{loss-error_stream[-2]:.3f}" if len(error_stream)>1 else None
                st.metric("Anomaly Score", f"{loss:.3f}", delta=delta_val, delta_color="inverse")
            with kpi_cols[2]:
                rul_display = f"{rul} cycles" if isinstance(rul, int) else str(rul)
                st.metric("RUL Estimate", rul_display)

            # 5. Maintenance Logging
            if loss > CRITICAL_THRESHOLD * 0.7:
                severity = "CRITICAL" if loss > CRITICAL_THRESHOLD else "WARNING"
                log_entry = f"{datetime.now().strftime('%H:%M:%S')} | {severity} | Score: {loss:.2f} | RUL: {rul_display}"
                logs.append(log_entry)
                log_expander.write(log_entry)

            # 6. Visualization
            with chart_holder.container():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                
                # Plot Raw Sensors
                ax1.plot([d[0] for d in raw_data_stream], color='#1A5276', label='Vibration')
                ax1.plot([d[1]/100 for d in raw_data_stream], color='#D35400', label='Temp (scaled)')
                ax1.set_title("Live Sensor Feeds")
                ax1.legend(loc='upper left')
                ax1.grid(alpha=0.3)
                
                # Plot Anomaly Score
                ax2.plot(error_stream, color='#C0392B', linewidth=2, label='Anomaly Score')
                ax2.axhline(y=CRITICAL_THRESHOLD, color='black', linestyle='--', label='Critical Limit')
                ax2.fill_between(range(len(error_stream)), error_stream, CRITICAL_THRESHOLD, 
                                 where=(np.array(error_stream) >= CRITICAL_THRESHOLD*0.7), 
                                 color='orange', alpha=0.3)
                ax2.set_title("System Health Index (LSTM Reconstruction Error)")
                ax2.legend(loc='upper left')
                ax2.grid(alpha=0.3)
                
                st.pyplot(fig)
                plt.close()

            # Break if failure reached
            if loss > CRITICAL_THRESHOLD:
                st.error("🚨 EQUIPMENT FAILURE DETECTED. Emergency shutdown initiated.")
                # Save Log to CSV
                df_log = pd.DataFrame(logs, columns=["Event Details"])
                df_log.to_csv("maintenance_report.csv", index=False)
                st.sidebar.download_button("📂 Download Failure Report", "maintenance_report.csv")
                break
        
        time.sleep(0.05)

else:
    st.info("👈 Adjust your equipment settings and click 'Start Live Stream' to begin monitoring.")
