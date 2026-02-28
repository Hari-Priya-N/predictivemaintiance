Import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# --- Page Configuration ---
st.set_page_config(page_title="SentinAI: Industrial Intelligence", layout="wide")

# ---------------------------------
# 1. AI Architecture (LSTM Autoencoder)
# ---------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, 16, batch_first=True)
        self.decoder = nn.LSTM(16, n_features, batch_first=True)
        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        x = hidden.repeat(self.seq_len, 1, 1).transpose(0, 1)
        x, _ = self.decoder(x)
        return self.output_layer(x)

@st.cache_resource
def init_system():
    """Initializes AI model and data scaler."""
    model = LSTMAutoencoder(12, 2)
    scaler = MinMaxScaler()
    # Fit scaler on a broader range for stability
    scaler.fit(np.array([[0.2, 40], [1.5, 80]]))
    return model, scaler

# ---------------------------------
# 2. Navigation & Sidebar UI
# ---------------------------------
model, scaler = init_system()
criterion = nn.MSELoss()

# Initialize session state for logs
if 'maintenance_logs' not in st.session_state:
    st.session_state.maintenance_logs = []

with st.sidebar:
    st.title("🏭 SentinAI Pro")
    st.markdown("---")
    page = st.radio("Navigation", ["🚀 Live Monitor", "📊 Deep Analytics", "📋 Maintenance Log"])
    st.markdown("---")
    st.header("Settings")
    run_sim = st.button("▶️ Start Stream", use_container_width=True)
    fail_trigger = st.toggle("Simulate Bearing Wear")
    threshold = st.slider("Anomaly Threshold (%)", 1.0, 10.0, 4.5)

# ---------------------------------
# 3. Page Routing Logic
# ---------------------------------

if page == "🚀 Live Monitor":
    st.header("Real-Time Asset Telemetry")
    
    # KPI Gauges Row
    col_v, col_t, col_a = st.columns(3)
    chart_slot = st.empty()
    
    if run_sim:
        data_log = []
        error_log = []
        
        for t in range(500):
            # Physics Simulation
            v = 0.5 + 0.05 * np.sin(t/5) + np.random.normal(0, 0.02)
            temp = 52 + 0.1 * np.cos(t/8) + np.random.normal(0, 0.1)
            
            if fail_trigger and t > 50:
                creep = np.exp((t-50)/70) * 0.04
                v += creep
                temp += creep * 15
                
            data_log.append([v, temp])
            
            # AI Inference
            if len(data_log) > 12:
                input_data = scaler.transform(np.array(data_log[-12:]))
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    recon = model(input_tensor)
                    loss = criterion(recon, input_tensor).item() * 100
                    error_log.append(loss)
                
                # Metrics Display
                col_v.metric("Vibration", f"{v:.2f}g")
                col_t.metric("Temperature", f"{temp:.1f}°C")
                col_a.metric("Anomaly Score", f"{loss:.2f}%", delta=f"{loss-threshold:.2f}%" if loss > threshold else None, delta_color="inverse")

                # Live Charts
                with chart_slot.container():
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                    fig.add_trace(go.Scatter(y=[d[0] for d in data_log], name="Vibration", line=dict(color='#3498DB')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=[d[1] for d in data_log], name="Temp", line=dict(color='#E67E22')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=error_log, name="AI Risk", fill='tozeroy', line=dict(color='#E74C3C')), row=2, col=1)
                    fig.update_layout(height=450, template="plotly_white", showlegend=False, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                # Logging Logic
                if loss > threshold:
                    log_entry = {"Time": datetime.now().strftime("%H:%M:%S"), "Status": "CRITICAL", "Score": round(loss, 2)}
                    if not st.session_state.maintenance_logs or st.session_state.maintenance_logs[-1]["Score"] != log_entry["Score"]:
                        st.session_state.maintenance_logs.append(log_entry)
            
            time.sleep(0.01)
    else:
        st.info("Click 'Start Stream' to begin monitoring.")

elif page == "📊 Deep Analytics":
    st.header("Diagnostic Phase-Space & Correlation")
    if len(st.session_state.maintenance_logs) > 0:
        # Note: In a real app, you'd pass the actual dataframe here. 
        # For the demo, we use simulated history.
        t_vals = np.linspace(0, 10, 100)
        v_vals = 0.5 + 0.1*np.random.randn(100)
        temp_vals = 50 + 2*np.random.randn(100)
        df_analysis = pd.DataFrame({"Vibration": v_vals, "Temperature": temp_vals})
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.imshow(df_analysis.corr(), text_auto=True, title="Feature Correlation"), use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(df_analysis, x="Vibration", y="Temperature", title="Cluster Analysis"), use_container_width=True)
    else:
        st.warning("No data captured yet. Run 'Live Monitor' first.")

elif page == "📋 Maintenance Log":
    st.header("System Event History")
    if st.session_state.maintenance_logs:
        df_logs = pd.DataFrame(st.session_state.maintenance_logs)
        st.dataframe(df_logs, use_container_width=True)
        st.download_button("Export Report (CSV)", df_logs.to_csv(index=False), "maintenance_report.csv")
    else:
        st.info("System healthy. No maintenance tickets generated.")
Requirement.txt for above
