import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Page Configuration ---
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

# ---------------------------------
# 1. Core Logic Functions
# ---------------------------------

@st.cache_data
def generate_sensor_data(n_samples=1000, n_anomalies=30):
    """Simulates conveyor belt sensor data."""
    rng = np.random.RandomState(42)
    normal_data = rng.randn(n_samples, 2)
    
    # Normal Operating Range
    vibration = 0.5 * normal_data[:, 0] + 0.1 * rng.rand(n_samples)
    temperature = 2 * normal_data[:, 1] + 50 + 0.1 * rng.rand(n_samples)
    
    # Anomalous Range
    anomalies_vibration = 2 * rng.rand(n_anomalies, 1) + 2.5
    anomalies_temperature = 2 * rng.rand(n_anomalies, 1) + 55
    
    X = np.vstack([
        np.column_stack([vibration, temperature]), 
        np.column_stack([anomalies_vibration, anomalies_temperature])
    ])
    labels = np.zeros(n_samples + n_anomalies)
    labels[n_samples:] = 1 
    
    df = pd.DataFrame(X, columns=['Vibration', 'Temperature'])
    df['is_anomaly'] = labels
    return df

def gan_augment_data(df, augmentation_ratio=0.3):
    """Generates synthetic normal data using a GAN."""
    X = df[['Vibration', 'Temperature']]
    latent_dim, input_dim = 10, X.shape[1]
    
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 32), nn.ReLU(),
                nn.Linear(32, input_dim)
            )
        def forward(self, z): return self.model(z)
    
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        def forward(self, x): return self.model(x)

    G, D = Generator(), Discriminator()
    criterion = nn.BCELoss()
    opt_G, opt_D = optim.Adam(G.parameters(), lr=0.001), optim.Adam(D.parameters(), lr=0.001)
    real_data = torch.tensor(X.values, dtype=torch.float32)

    # Training Loop
    for _ in range(150):
        opt_D.zero_grad()
        z = torch.randn(real_data.size(0), latent_dim)
        fake_data = G(z)
        loss_D = criterion(D(real_data), torch.ones(real_data.size(0), 1)) + \
                 criterion(D(fake_data.detach()), torch.zeros(real_data.size(0), 1))
        loss_D.backward(); opt_D.step()
        
        opt_G.zero_grad()
        loss_G = criterion(D(fake_data), torch.ones(real_data.size(0), 1))
        loss_G.backward(); opt_G.step()

    n_synthetic = int(len(X) * augmentation_ratio)
    synthetic_data = G(torch.randn(n_synthetic, latent_dim)).detach().numpy()
    synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)
    synthetic_df['is_anomaly'] = 0
    return pd.concat([df, synthetic_df], ignore_index=True)

def train_anomaly_models(df, contam=0.03):
    """Trains 4 models and creates an ensemble urgency score."""
    X = df[['Vibration', 'Temperature']]
    
    # Models
    df['pred_if'] = IsolationForest(contamination=contam, random_state=42).fit_predict(X)
    df['pred_ocsvm'] = OneClassSVM(kernel='rbf', nu=contam).fit_predict(X)
    df['pred_lof'] = LocalOutlierFactor(n_neighbors=20, contamination=contam).fit_predict(X)
    df['pred_dbscan'] = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    
    # Risk Score (0-4)
    df['risk_score'] = (
        (df['pred_if'] == -1).astype(int) + (df['pred_ocsvm'] == -1).astype(int) +
        (df['pred_lof'] == -1).astype(int) + (df['pred_dbscan'] == -1).astype(int)
    )
    
    def get_urgency(s):
        if s >= 3: return "HIGH"
        if s == 2: return "MEDIUM"
        if s == 1: return "LOW"
        return "NORMAL"
    
    df['urgency'] = df['risk_score'].apply(get_urgency)
    return df

# ---------------------------------
# 2. Streamlit Dashboard UI
# ---------------------------------

st.title("🏭 Predictive Maintenance AI Dashboard")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Configuration")
n_samples = st.sidebar.slider("Sensor Samples", 500, 2000, 1000)
n_anoms = st.sidebar.slider("Simulated Failures", 10, 100, 30)
aug_ratio = st.sidebar.slider("GAN Data Augmentation (%)", 10, 100, 30) / 100
contam_rate = st.sidebar.slider("Detection Sensitivity", 0.01, 0.1, 0.03)

if st.sidebar.button("Run Diagnostics"):
    # Phase 1: Original Data
    raw_data = generate_sensor_data(n_samples, n_anoms)
    results_orig = train_anomaly_models(raw_data.copy(), contam_rate)
    
    # Phase 2: GAN Augmentation
    with st.spinner("Generating synthetic baseline with GAN..."):
        augmented_data = gan_augment_data(raw_data, aug_ratio)
        results_aug = train_anomaly_models(augmented_data.copy(), contam_rate)

    # --- Section 1: Maintenance Alerts ---
    st.header("🚨 Real-Time Maintenance Tickets")
    alerts = results_orig[results_orig['risk_score'] >= 1].sort_values(by='risk_score', ascending=False)
    
    if not alerts.empty:
        # Create columns for cards
        cols = st.columns(3)
        for i, (idx, row) in enumerate(alerts.head(9).iterrows()):
            with cols[i % 3]:
                color = "red" if row['urgency'] == "HIGH" else "orange" if row['urgency'] == "MEDIUM" else "blue"
                st.info(f"**Index {int(idx)}**\n\nPriority: :{color}[{row['urgency']}]\n\n"
                        f"Vib: {row['Vibration']:.2f} | Temp: {row['Temperature']:.2f}\n\n"
                        f"Risk Score: {int(row['risk_score'])}/4")
    else:
        st.success("All systems operating within normal parameters.")

    # --- Section 2: Model Performance ---
    st.markdown("---")
    st.header("📊 Ensemble Leaderboard (F1-Score)")
    
    performance_data = []
    for state, res in [("Original", results_orig), ("Augmented", results_aug)]:
        y_true = res['is_anomaly']
        for m in ['if', 'ocsvm', 'lof', 'dbscan']:
            y_pred = (res[f'pred_{m}'] == -1).astype(int)
            performance_data.append({
                "Model": m.upper(),
                "Dataset": state,
                "F1-Score": f1_score(y_true, y_pred, zero_division=0)
            })
    
    perf_df = pd.DataFrame(performance_data)
    
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
        sns.barplot(data=perf_df, x='Model', y='F1-Score', hue='Dataset', palette='mako', ax=ax_perf)
        st.pyplot(fig_perf)
    with col_table:
        st.dataframe(perf_df.pivot(index="Model", columns="Dataset", values="F1-Score"), use_container_width=True)

    # --- Section 3: Spatial Mapping ---
    st.markdown("---")
    st.header("🔍 Spatial Anomaly Distribution")
    
    tab1, tab2 = st.tabs(["Original Data", "GAN-Augmented Data"])
    
    with tab1:
        fig_orig, ax_orig = plt.subplots(figsize=(12, 5))
        sns.scatterplot(data=results_orig, x='Vibration', y='Temperature', 
                        hue='urgency', palette='coolwarm', style='is_anomaly', ax=ax_orig)
        st.pyplot(fig_orig)
        
    with tab2:
        fig_aug, ax_aug = plt.subplots(figsize=(12, 5))
        sns.scatterplot(data=results_aug, x='Vibration', y='Temperature', 
                        hue='urgency', palette='coolwarm', style='is_anomaly', ax=ax_aug)
        st.pyplot(fig_aug)

else:
    st.info("👈 Use the control panel on the left to configure the sensor simulation and run the AI models.")
