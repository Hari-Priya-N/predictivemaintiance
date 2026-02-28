import streamlit as st
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
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, fftfreq

# --- Page Configuration ---
st.set_page_config(page_title="SentinAI Pro: Fragmented Intelligence", layout="wide", page_icon="⚙️")

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
    model = LSTMAutoencoder(12, 2)
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0.2, 40], [1.5, 80]]))
    return model, scaler

# ---------------------------------
# 2. Initialization & Sidebar
# ---------------------------------
model, scaler = init_system()
criterion = nn.MSELoss()

# Initialize session state for persistence across fragments
if 'maintenance_logs' not in st.session_state:
    st.session_state.maintenance_logs = []
if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

with st.sidebar:
    st.title("⚙️ SentinAI v3.1")
    st.markdown("---")
    page = st.radio("Navigation", ["🚀 Live Monitor", "📋 Maintenance Log"])
    st.markdown("---")
    
    # Using session state for the button to keep it active during navigation
    if st.button("▶️ Start / Reset Stream", use_container_width=True):
        st.session_state.run_sim = True
        st.session_state.maintenance_logs = [] # Clear logs on new run
    
    if st.button("🛑 Stop Stream", use_container_width=True):
        st.session_state.run_sim = False
        
    fail_trigger = st.toggle("Simulate Bearing Wear")
    threshold = st.slider("Anomaly Threshold (%)", 1.0, 15.0, 6.0)

# ---------------------------------
# 3. Fragmented Simulation Logic
# ---------------------------------

@st.fragment
def run_live_monitor():
    """This fragment runs the simulation without locking the rest of the UI."""
    st.header("Digital Twin & Spectral Analysis")
    
    tab1, tab2 = st.tabs(["📉 Digital Twin Stream", "🔬 Spectral & Phase-Space"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        v_met, t_met, a_met = c1.empty(), c2.empty(), c3.empty()
        chart_slot = st.empty()
    
    with tab2:
        spectral_slot = st.empty()

    if st.session_state.run_sim:
        data_log, recon_log, error_log = [], [], []
        UPDATE_INTERVAL = 15 
        
        for t in range(2000):
            # If user stops via sidebar, break the loop
            if not st.session_state.run_sim:
                break
                
            # --- Physics Simulation ---
            v_base = 0.5 + 0.05 * np.sin(t/5) + np.random.normal(0, 0.01)
            t_base = 52 + 0.1 * np.cos(t/8) + np.random.normal(0, 0.05)
            
            if fail_trigger and t > 50:
                drift = np.exp((t-50)/150) * 0.02
                v_base += drift + (0.05 * np.sin(t*0.8) * drift)
                t_base += drift * 15
            
            data_log.append([v_base, t_base])
            
            # --- AI Inference ---
            if len(data_log) >= 12:
                window = np.array(data_log[-12:])
                input_scaled = scaler.transform(window)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                    loss = criterion(output_tensor, input_tensor).item() * 100
                    error_log.append(loss)
                    last_recon = scaler.inverse_transform(output_tensor.squeeze(0).numpy())[-1]
                    recon_log.append(last_recon)
                
                # --- Update Dashboard UI ---
                v_met.metric("Vibration", f"{v_base:.2f}g")
                t_met.metric("Temperature", f"{t_base:.1f}°C")
                a_met.metric("AI Risk", f"{loss:.1f}%")

                with chart_slot.container():
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                    fig.add_trace(go.Scatter(y=[d[0] for d in data_log[-300:]], name="Actual", line=dict(color='#00d1ff')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=[r[0] for r in recon_log[-300:]], name="AI", line=dict(color='white', dash='dot', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(y=error_log[-300:], name="Risk", fill='tozeroy', line=dict(color='#ff4b4b')), row=2, col=1)
                    fig.update_layout(height=400, template="plotly_dark", showlegend=False, margin=dict(t=5, b=5))
                    st.plotly_chart(fig, use_container_width=True, key=f"twin_{t}")

                # --- Throttled Spectral Analysis ---
                if t % UPDATE_INTERVAL == 0:
                    with spectral_slot.container():
                        m1, m2 = st.columns(2)
                        # FFT
                        n_fft = 128
                        if len(data_log) > n_fft:
                            yf = fft([d[0] for d in data_log[-n_fft:]])
                            xf = fftfreq(n_fft, 1)[:n_fft//2]
                            fig_fft = go.Figure(data=go.Bar(x=xf, y=2.0/n_fft * np.abs(yf[0:n_fft//2]), marker_color='#00ff9d'))
                            fig_fft.update_layout(height=300, title="Live FFT Spectrum", template="plotly_dark")
                            m1.plotly_chart(fig_fft, use_container_width=True, key=f"fft_{t}")
                        
                        # Phase Space
                        fig_p = px.scatter(pd.DataFrame(data_log[-200:], columns=["V","T"]), x="V", y="T")
                        fig_p.update_layout(height=300, title="Phase-Space Drift", template="plotly_dark")
                        m2.plotly_chart(fig_p, use_container_width=True, key=f"phase_{t}")

                # --- LOGGING ENGINE (Fix) ---
                if loss > threshold:
                    # Append directly to session state
                    new_log = {"Time": datetime.now().strftime("%H:%M:%S"), "Anomaly": f"{loss:.1f}%", "Status": "CRITICAL"}
                    # Only log if it's a new unique timestamp or significant change
                    if not st.session_state.maintenance_logs or st.session_state.maintenance_logs[-1]["Time"] != new_log["Time"]:
                        st.session_state.maintenance_logs.append(new_log)
            
            time.sleep(0.01)
    else:
        st.info("System Standby. Use the sidebar to start the AI ingestion engine.")

# ---------------------------------
# 4. Page Routing
# ---------------------------------
if page == "🚀 Live Monitor":
    run_live_monitor()

elif page == "📋 Maintenance Log":
    st.header("System Incident Archive")
    st.markdown("This log updates in real-time as the AI detects drift.")
    
    if st.session_state.maintenance_logs:
        df = pd.DataFrame(st.session_state.maintenance_logs)
        st.error(f"Total Critical Events: {len(df)}")
        st.dataframe(df.iloc[::-1], use_container_width=True) # Newest first
        
        # Download link
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV Report", data=csv, file_name="maintenance_log.csv", mime="text/csv")
    else:
        st.success("No anomalies detected. System health is optimal.")
