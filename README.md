 ⚙️ SentinAI Pro v3.1: Industrial Digital Twin - https://predictivemaintiance-pm3pbjvve497tzrjlczwad.streamlit.app/#spatial-anomaly-distribution

SentinAI Pro is a high-fidelity predictive maintenance dashboard that utilizes Deep Learning to monitor industrial assets. By combining LSTM Autoencoders with Spectral Analysis (FFT), it identifies mechanical "drift" and predicts the Remaining Useful Life (RUL) of machinery before a catastrophic breakdown occurs.

-

 🚀 Key Features

 🔹 AI-Powered Digital Twin

The system uses an LSTM Autoencoder to learn the "healthy heartbeat" of a machine. It compares live sensor data against an AI-generated "ideal" reconstruction. The difference between the two (the residual) serves as a highly sensitive anomaly score, identifying deviations that traditional threshold alerts miss.

 🔹 Spectral Intelligence (FFT)

The dashboard performs a Fast Fourier Transform on vibration data, converting messy time-domain waves into a frequency spectrum. This allows engineers to see specific "harmonic spikes" caused by bearing pits, imbalance, or misalignment.

 🔹 Predictive RUL & Trend Analysis

Using Linear Regression on the anomaly trend, the system calculates a "Time to Critical" metric. This shifts maintenance from a reactive "fix it when it breaks" model to a proactive "fix it before it fails" strategy.

 🔹 Fragmented UI Architecture

Built with Streamlit Fragments, the simulation engine runs independently of the UI navigation. Users can audit logs or export CSV reports without interrupting the live AI data stream or the physics simulation.

-

 🛠️ Technical Stack

 Core AI: PyTorch (LSTM Autoencoder)
 Signal Processing: SciPy (FFT Analysis)
 Data Science: NumPy, Pandas, Scikit-Learn
 Interface: Streamlit (Fragments & Session State)
 Visualization: Plotly (Digital Twin Overlay & Phase-Space)

-

 🚦 Getting Started

 1. Installation

```bash
git clone https://github.com/yourusername/sentinai-pro.git
cd sentinai-pro
pip install -r requirements.txt

```

 2. Launch the Dashboard

```bash
streamlit run app.py

```

-

 📖 How the Model Works

The project relies on Unsupervised Anomaly Detection:

1. Compression: The Encoder "squeezes" 12-point sensor sequences into a low-dimensional summary, forcing the AI to memorize the machine's core physical rhythm.
2. Reconstruction: The Decoder tries to rebuild the original signal from that summary.
3. Detection: Since the model is trained exclusively on "healthy" patterns, it cannot accurately rebuild "failure" patterns. This failure to reconstruct (measured via Mean Squared Error) triggers the maintenance alert.

-

 📋 Maintenance & Audit

The system includes an Incident Archive that automatically logs:

 Event Timestamp: Exact time of anomaly detection.
 Anomaly Severity: The percentage of deviation from the Digital Twin.
 Exportable Reports: One-click CSV downloads for maintenance teams and engineering audits.

-

