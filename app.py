import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler

# --- Page Config ---
st.set_page_config(page_title="SentinAI Pro: 360° Monitoring", layout="wide")

# ---------------------------------
# 1. AI Logic (LSTM Autoencoder)
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
    model = LSTMAutoencoder(12, 2)
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0.3, 40], [1.2, 75]]))
    return model, scaler

# ---------------------------------
# 2. Main UI & Layout
# ---------------------------------
st.title("🛡️ SentinAI: 360° Industrial Intelligence")
st.markdown("---")

model, scaler = init_system()
criterion = nn.MSELoss()

# Sidebar
st.sidebar.header("🕹️ Control Center")
run_sim = st.sidebar.button("▶️ Start Live Telemetry", use_container_width=True)
fail_trigger = st.sidebar.toggle("Simulate Bearing Fatigue")
st.sidebar.markdown("---")
diag_expander = st.sidebar.expander("AI Diagnostic Details")
diag_expander.write("Model: LSTM Autoencoder")
diag_expander.write("Input Window: 12 cycles")

# Layout: Gauges on top, Charts in middle, Diagnostics on bottom
col_g1, col_g2, col_g3 = st.columns(3)
main_chart_slot = st.empty()
col_diag1, col_diag2 = st.columns(2)

# ---------------------------------
# 3. Live Execution Loop
# ---------------------------------
if run_sim:
    data_log = []
    error_log = []
    
    for t in range(400):
        # 1. Physics Data Simulation
        v = 0.5 + 0.05 * np.sin(t/5) + np.random.normal(0, 0.02)
        temp = 52 + 0.1 * np.cos(t/8) + np.random.normal(0, 0.1)
        
        if fail_trigger and t > 50:
            creep = np.exp((t-50)/60) * 0.03
            v += creep + np.random.normal(0, 0.01)
            temp += creep * 14
            
        data_log.append([v, temp])
        df_current = pd.DataFrame(data_log, columns=["Vibration", "Temperature"])
        
        # 2. KPI Gauges
        col_g1.metric("Vibration (g)", f"{v:.2f}", delta=f"{v-0.5:.2f}" if t>0 else None, delta_color="inverse")
        col_g2.metric("Temperature (°C)", f"{temp:.1f}", delta=f"{temp-52:.1f}" if t>0 else None, delta_color="inverse")
        
        # 3. AI Processing
        if len(data_log) > 12:
            input_data = scaler.transform(np.array(data_log[-12:]))
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                recon = model(input_tensor)
                loss = criterion(recon, input_tensor).item() * 100
                error_log.append(loss)
            
            col_g3.metric("Anomaly Score", f"{loss:.2f}%", delta=f"{loss-1:.2f}%" if len(error_log)>1 else None, delta_color="inverse")

            # 4. Main Time-Series Visual (Plotly)
            with main_chart_slot.container():
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                   subplot_titles=("Live Telemetry", "AI Reconstruction Loss"))
                fig.add_trace(go.Scatter(y=df_current["Vibration"], name="Vibration", line=dict(color='#3498DB')), row=1, col=1)
                fig.add_trace(go.Scatter(y=df_current["Temperature"], name="Temperature", line=dict(color='#E67E22')), row=1, col=1)
                fig.add_trace(go.Scatter(y=error_log, name="Anomaly Score", fill='tozeroy', line=dict(color='#E74C3C')), row=2, col=1)
                fig.update_layout(height=450, template="plotly_white", margin=dict(t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

            # 5. Diagnostic Visuals (Heatmap & Scatter)
            with col_diag1:
                # Correlation Heatmap
                corr = df_current.corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', 
                                    title="Feature Correlation Matrix")
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col_diag2:
                # Spatial Anomaly Distribution (Scatter)
                # Color points by anomaly score (recent)
                scatter_colors = [0]*12 + error_log
                fig_scatter = px.scatter(df_current, x="Vibration", y="Temperature", 
                                        color=scatter_colors, color_continuous_scale='viridis',
                                        title="Phase Space Analysis (Vibration vs Temp)")
                fig_scatter.update_layout(height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)

        time.sleep(0.01)
else:
    st.info("System Ready. Click **Start Live Telemetry** to monitor asset phase-space and correlations.")
