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
st.set_page_config(page_title="SentinAI Pro: Spectral Intelligence", layout="wide", page_icon="⚙️")

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

if 'maintenance_logs' not in st.session_state:
    st.session_state.maintenance_logs = []

with st.sidebar:
    st.title("⚙️ SentinAI v3.0")
    st.markdown("---")
    page = st.radio("Navigation", ["🚀 Live Monitor", "📋 Maintenance Log"])
    st.markdown("---")
    run_sim = st.button("▶️ Start Ingestion", use_container_width=True)
    fail_trigger = st.toggle("Simulate Bearing Wear")
    threshold = st.slider("Anomaly Threshold (%)", 1.0, 15.0, 6.0)
    
    st.markdown("### Predictive Insights")
    rul_slot = st.empty() 
    st.info("The FFT panel (Tab 2) shows frequency spikes. Watch for high-frequency growth during failure simulation.")

# ---------------------------------
# 3. Live Monitor Page
# ---------------------------------
if page == "🚀 Live Monitor":
    st.header("Digital Twin & Spectral Analysis")
    
    tab1, tab2 = st.tabs(["📉 Digital Twin Stream", "🔬 Spectral & Phase-Space"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        v_met = c1.empty()
        t_met = c2.empty()
        a_met = c3.empty()
        chart_slot = st.empty()
    
    with tab2:
        # FFT and Correlation layout
        spectral_slot = st.empty()

    if run_sim:
        data_log = []     
        recon_log = []    
        error_log = []    
        
        UPDATE_INTERVAL = 15 # Throttling for UI smoothness
        
        for t in range(1500):
            # --- Physics Simulation ---
            # Normal: Low frequency sine waves
            v_base = 0.5 + 0.05 * np.sin(t/5) + np.random.normal(0, 0.01)
            t_base = 52 + 0.1 * np.cos(t/8) + np.random.normal(0, 0.05)
            
            if fail_trigger and t > 50:
                drift = np.exp((t-50)/100) * 0.02
                v_base += drift
                # Adding high-frequency jitter to vibration (the 'bearing squeal')
                v_base += 0.05 * np.sin(t*0.8) * drift 
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
                
                # --- RUL Prediction ---
                if len(error_log) > 30:
                    y_trend = np.array(error_log[-30:]).reshape(-1, 1)
                    x_trend = np.arange(30).reshape(-1, 1)
                    reg = LinearRegression().fit(x_trend, y_trend)
                    slope = reg.coef_[0][0]
                    if slope > 0.01:
                        cycles_to_fail = max(0, (threshold*2.5 - loss) / slope)
                        rul_slot.metric("Estimated Cycles to Failure", f"{int(cycles_to_fail)}")
                    else:
                        rul_slot.metric("System Health", "STABLE")

                # --- Update Metrics ---
                v_met.metric("Vibration", f"{v_base:.2f}g")
                t_met.metric("Temperature", f"{t_base:.1f}°C")
                a_met.metric("AI Risk", f"{loss:.1f}%")

                # --- Tab 1: Digital Twin (High Speed) ---
                with chart_slot.container():
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Scatter(y=[d[0] for d in data_log], name="Actual", line=dict(color='#00d1ff', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(y=[r[0] for r in recon_log], name="AI Expected", line=dict(color='rgba(255,255,255,0.3)', dash='dot')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=error_log, name="Risk", fill='tozeroy', line=dict(color='#ff4b4b')), row=2, col=1)
                    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)
                    fig.update_layout(height=450, template="plotly_dark", margin=dict(t=5, b=5, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"t1_{t}")

                # --- Tab 2: Spectral & Matrix (Throttled) ---
                if t % UPDATE_INTERVAL == 0:
                    with spectral_slot.container():
                        m1, m2 = st.columns(2)
                        
                        # Calculate FFT for Vibration
                        # We take the last 128 points for a clean spectrum
                        n_fft = 128
                        if len(data_log) > n_fft:
                            v_series = [d[0] for d in data_log[-n_fft:]]
                            yf = fft(v_series)
                            xf = fftfreq(n_fft, 1)[:n_fft//2]
                            mag = 2.0/n_fft * np.abs(yf[0:n_fft//2])
                            
                            fig_fft = go.Figure(data=go.Bar(x=xf, y=mag, marker_color='#00ff9d'))
                            fig_fft.update_layout(height=350, title="FFT (Frequency Spectrum)", template="plotly_dark", 
                                                 xaxis_title="Frequency", yaxis_title="Magnitude", margin=dict(t=40, b=10))
                            m1.plotly_chart(fig_fft, use_container_width=True, key=f"fft_{t}")
                        
                        # Phase Space Scatter
                        df_p = pd.DataFrame(data_log[-300:], columns=["V", "T"])
                        fig_p = px.scatter(df_p, x="V", y="T", color_discrete_sequence=['#ffcc00'])
                        fig_p.update_layout(height=350, title="Phase-Space Coupling", template="plotly_dark", margin=dict(t=40, b=10))
                        m2.plotly_chart(fig_p, use_container_width=True, key=f"phase_{t}")

                if loss > threshold:
                    st.session_state.maintenance_logs.append({"Time": datetime.now().strftime("%H:%M:%S"), "Anomaly": f"{loss:.1f}%"})
            
            time.sleep(0.04)
    else:
        st.info("System Standby. Click 'Start Ingestion' to begin AI Digital Twin monitoring.")

# ---------------------------------
# 4. Maintenance Log
# ---------------------------------
elif page == "📋 Maintenance Log":
    st.header("Incident Archive")
    if st.session_state.maintenance_logs:
        st.dataframe(pd.DataFrame(st.session_state.maintenance_logs).iloc[::-1], use_container_width=True)
    else:
        st.success("No anomalies detected in the current cycle.")
