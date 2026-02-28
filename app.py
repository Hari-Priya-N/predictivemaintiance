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

# --- Page Configuration ---
st.set_page_config(page_title="SentinAI: Industrial Intelligence", layout="wide", page_icon="🏭")

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
# 2. Initialization & Sidebar
# ---------------------------------
model, scaler = init_system()
criterion = nn.MSELoss()

if 'maintenance_logs' not in st.session_state:
    st.session_state.maintenance_logs = []

with st.sidebar:
    st.title("🏭 SentinAI Pro")
    st.markdown("---")
    page = st.radio("Navigation", ["🚀 Live Monitor", "📋 Maintenance Log"])
    st.markdown("---")
    st.header("Simulation Control")
    run_sim = st.button("▶️ Start Real-Time Stream", use_container_width=True)
    fail_trigger = st.toggle("Simulate Bearing Wear (Anomaly)")
    threshold = st.slider("Anomaly Threshold (%)", 1.0, 10.0, 4.5)
    st.info("The AI detects deviations from the 'normal' physics-based sine wave patterns.")

# ---------------------------------
# 3. Live Monitor Page (With Multi-Tab Updates)
# ---------------------------------
if page == "🚀 Live Monitor":
    st.header("Real-Time Asset Telemetry")
    
    # Define Layout Containers
    tab1, tab2 = st.tabs(["📈 Live Streams", "🔬 Diagnostic Matrix"])
    
    with tab1:
        col_v, col_t, col_a = st.columns(3)
        chart_slot = st.empty()
    
    with tab2:
        matrix_slot = st.empty()

    if run_sim:
        data_log = []
        error_log = []
        
        # Simulation Loop
        for t in range(1000):
            # 1. Physics Simulation (Normal vs. Anomaly)
            v = 0.5 + 0.05 * np.sin(t/5) + np.random.normal(0, 0.02)
            temp = 52 + 0.1 * np.cos(t/8) + np.random.normal(0, 0.1)
            
            if fail_trigger and t > 30:
                creep = np.exp((t-30)/60) * 0.03
                v += creep
                temp += creep * 12
                
            data_log.append([v, temp])
            
            # 2. AI Inference (Windowed)
            if len(data_log) > 12:
                input_data = scaler.transform(np.array(data_log[-12:]))
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    recon = model(input_tensor)
                    loss = criterion(recon, input_tensor).item() * 100
                    error_log.append(loss)
                
                # --- UPDATE TAB 1: Metrics & Charts ---
                col_v.metric("Vibration", f"{v:.2f}g")
                col_t.metric("Temperature", f"{temp:.1f}°C")
                col_a.metric("Anomaly Score", f"{loss:.2f}%", 
                           delta=f"{loss-threshold:.2f}%" if loss > threshold else None, 
                           delta_color="inverse")

                with chart_slot.container():
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                    fig.add_trace(go.Scatter(y=[d[0] for d in data_log], name="Vibration", line=dict(color='#3498DB', width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(y=[d[1] for d in data_log], name="Temperature", line=dict(color='#E67E22', width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(y=error_log, name="AI Risk Score", fill='tozeroy', line=dict(color='#E74C3C')), row=2, col=1)
                    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)
                    fig.update_layout(height=450, template="plotly_white", margin=dict(t=10, b=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"stream_{t}")

                # --- UPDATE TAB 2: Live Matrix & Phase-Space ---
                with matrix_slot.container():
                    df_live = pd.DataFrame(data_log, columns=["Vibration", "Temperature"])
                    m_col1, m_col2 = st.columns(2)
                    
                    with m_col1:
                        # Heatmap for Correlation
                        fig_corr = px.imshow(df_live.corr(), text_auto=".2f", 
                                            color_continuous_scale='RdBu_r', 
                                            title="Real-Time Feature Correlation")
                        st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{t}")
                    
                    with m_col2:
                        # Phase Space Scatter (Visualizing Failure Clusters)
                        fig_scatter = px.scatter(df_live, x="Vibration", y="Temperature", 
                                                title="Vibration vs. Temperature Phase-Space",
                                                color_discrete_sequence=['#8E44AD'])
                        st.plotly_chart(fig_scatter, use_container_width=True, key=f"scat_{t}")

                # 3. Critical Alert Logging
                if loss > threshold:
                    log_entry = {"Time": datetime.now().strftime("%H:%M:%S"), "Status": "CRITICAL", "Score": round(loss, 2)}
                    # Prevent duplicate log spam
                    if not st.session_state.maintenance_logs or st.session_state.maintenance_logs[-1]["Score"] != log_entry["Score"]:
                        st.session_state.maintenance_logs.append(log_entry)
            
            time.sleep(0.05)
    else:
        st.info("System Standby. Click 'Start Real-Time Stream' to begin data ingestion.")

# ---------------------------------
# 4. Maintenance Log Page
# ---------------------------------
elif page == "📋 Maintenance Log":
    st.header("System Event History")
    if st.session_state.maintenance_logs:
        df_logs = pd.DataFrame(st.session_state.maintenance_logs)
        st.error(f"⚠️ {len(df_logs)} Anomaly events detected in current session.")
        st.dataframe(df_logs.iloc[::-1], use_container_width=True) # Show newest first
        st.download_button("Export Incident Report", df_logs.to_csv(index=False), "maintenance_report.csv")
    else:
        st.success("All systems nominal. No anomalies detected.")
