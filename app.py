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
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# ---------------------------------
# 1. Logic Functions (Cached for Performance)
# ---------------------------------

@st.cache_data
def generate_sensor_data(n_samples=1000, n_anomalies=30):
    rng = np.random.RandomState(42)
    normal_data = rng.randn(n_samples, 2)
    vibration = 0.5 * normal_data[:, 0] + 0.1 * rng.rand(n_samples)
    temperature = 2 * normal_data[:, 1] + 50 + 0.1 * rng.rand(n_samples)
    anomalies_vibration = 2 * rng.rand(n_anomalies, 1) + 2.5
    anomalies_temperature = 2 * rng.rand(n_anomalies, 1) + 55
    X = np.vstack([np.column_stack([vibration, temperature]), np.column_stack([anomalies_vibration, anomalies_temperature])])
    labels = np.zeros(n_samples + n_anomalies)
    labels[n_samples:] = 1 
    df = pd.DataFrame(X, columns=['Vibration', 'Temperature'])
    df['is_anomaly'] = labels
    return df

def gan_augment_data(df, augmentation_ratio=0.3):
    X = df[['Vibration', 'Temperature']]
    latent_dim, input_dim = 10, X.shape[1]
    
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim))
        def forward(self, z): return self.model(z)
    
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        def forward(self, x): return self.model(x)

    G, D = Generator(), Discriminator()
    criterion = nn.BCELoss()
    opt_G, opt_D = optim.Adam(G.parameters(), lr=0.001), optim.Adam(D.parameters(), lr=0.001)
    real_data = torch.tensor(X.values, dtype=torch.float32)

    # Simplified training loop for UI responsiveness
    for _ in range(100):
        opt_D.zero_grad()
        z = torch.randn(real_data.size(0), latent_dim)
        fake_data = G(z)
        loss_D = criterion(D(real_data), torch.ones(real_data.size(0), 1)) + criterion(D(fake_data.detach()), torch.zeros(real_data.size(0), 1))
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
    X = df[['Vibration', 'Temperature']]
    df['pred_if'] = IsolationForest(contamination=contam, random_state=42).fit_predict(X)
    df['pred_ocsvm'] = OneClassSVM(kernel='rbf', nu=contam).fit_predict(X)
    df['pred_lof'] = LocalOutlierFactor(n_neighbors=20, contamination=contam).fit_predict(X)
    df['pred_dbscan'] = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    
    df['risk_score'] = ((df['pred_if'] == -1).astype(int) + (df['pred_ocsvm'] == -1).astype(int) +
                        (df['pred_lof'] == -1).astype(int) + (df['pred_dbscan'] == -1).astype(int))
    
    def get_urgency(s):
        return "HIGH" if s >= 3 else ("MEDIUM" if s == 2 else ("LOW" if s == 1 else "NORMAL"))
    df['urgency'] = df['risk_score'].apply(get_urgency)
    return df

# ---------------------------------
# 2. UI Layout
# ---------------------------------

st.title("🏭 AI Predictive Maintenance Dashboard")
st.markdown("Monitor conveyor belt health using a GAN-augmented anomaly detection ensemble.")

# Sidebar Controls
st.sidebar.header("Control Panel")
n_samples = st.sidebar.slider("Sample Size", 500, 2000, 1000)
n_anoms = st.sidebar.slider("True Anomalies", 10, 100, 30)
aug_ratio = st.sidebar.slider("GAN Augmentation Ratio", 0.1, 1.0, 0.3)
contam_rate = st.sidebar.slider("Model Contamination Rate", 0.01, 0.1, 0.03)

if st.sidebar.button("Run Pipeline"):
    # Phase 1: Data
    data = generate_sensor_data(n_samples, n_anoms)
    results_orig = train_anomaly_models(data.copy(), contam_rate)
    
    with st.spinner("GAN is hallucinating normal data..."):
        augmented_data = gan_augment_data(data, aug_ratio)
        results_aug = train_anomaly_models(augmented_data.copy(), contam_rate)

    # --- Metrics Section ---
    st.header("📊 Performance Leaderboard")
    col1, col2 = st.columns([2, 1])
    
    metrics = []
    for name, d, state in [('IF', results_orig, 'Original'), ('OCSVM', results_orig, 'Original'), 
                           ('LOF', results_orig, 'Original'), ('DBSCAN', results_orig, 'Original'),
                           ('IF', results_aug, 'Augmented'), ('OCSVM', results_aug, 'Augmented'), 
                           ('LOF', results_aug, 'Augmented'), ('DBSCAN', results_aug, 'Augmented')]:
        y_true = d['is_anomaly']
        y_pred = (d[f'pred_{name.lower()}'] == -1).astype(int)
        metrics.append({'Model': name, 'Dataset': state, 'F1': f1_score(y_true, y_pred, zero_division=0)})
    
    metric_df = pd.DataFrame(metrics)
    
    with col1:
        fig_f1, ax_f1 = plt.subplots()
        sns.barplot(data=metric_df, x='Model', y='F1', hue='Dataset', palette='viridis', ax=ax_f1)
        st.pyplot(fig_f1)
    
    with col2:
        st.dataframe(metric_df.sort_values(by='F1', ascending=False), hide_index=True)

    # --- Alerts Section ---
    st.header("🚨 Active Maintenance Tickets")
    alerts = results_orig[results_orig['risk_score'] >= 1].sort_values(by='risk_score', ascending=False)
    
    if not alerts.empty:
        for _, row in alerts.iterrows():
            color = "red" if row['urgency'] == "HIGH" else "orange" if row['urgency'] == "MEDIUM" else "blue"
            st.markdown(f":{color}[**{row['urgency']} PRIORITY**] - Index {int(_)}: Vibration: {row['Vibration']:.2f}, Temp: {row['Temperature']:.2f} (Score: {int(row['risk_score'])}/4)")
    else:
        st.success("All systems nominal. No alerts found.")

    # --- Visualization Section ---
    st.header("🔍 Spatial Anomaly Mapping")
    tab1, tab2 = st.tabs(["Isolation Forest", "One-Class SVM"])
    
    palette = {'Normal': 'skyblue', 'Anomaly': 'red'}
    
    with tab1:
        fig_map, ax_map = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(data=results_orig, x='Vibration', y='Temperature', hue='pred_if', palette='coolwarm', ax=ax_map[0])
        ax_map[0].set_title("Original Data Detections")
        sns.scatterplot(data=results_aug, x='Vibration', y='Temperature', hue='pred_if', palette='coolwarm', ax=ax_map[1])
        ax_map[1].set_title("Augmented Data Detections")
        st.pyplot(fig_map)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Pipeline' to begin.")
