import streamlit as st
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Bead Mill Digital Twin | Process Engineering", 
    layout="wide"
)

@st.cache_resource
def load_digital_twin():
    """Rebuilds the architecture and loads the trained PyTorch weights."""
    # 4 Inputs (x, y, t, RPM) -> 3 Hidden Layers -> 5 Outputs (u, v, p, phi, D90)
    net = dde.nn.FNN([4] + [40] * 3 + [5], "tanh", "Glorot normal")
    model = dde.Model(data=None, net=net)
    
    weights_path = "models/beadmill_pinn_weights.pt"
    if os.path.exists(weights_path):
        model.restore(weights_path, verbose=0)
        return model
    else:
        return None

model = load_digital_twin()

st.title("Bead Mill Digital Twin: API Particle Size Predictor")
st.markdown("Physics-Informed Neural Network (PINN) inference for Prednisolone scale-up.")

with st.sidebar:
    st.header("Process Parameters")
    target_rpm = st.slider("Agitator Speed (RPM)", 500.0, 3000.0, 1500.0, step=100.0)
    grinding_time = st.slider("Grinding Time (minutes)", 0.0, 120.0, 60.0, step=5.0)
    run_sim = st.button("Simulate Batch", type="primary", use_container_width=True)

if run_sim:
    if model is None:
        st.error("Model weights not found. Please ensure 'beadmill_pinn_weights.pt' is in the 'models/' directory.")
    else:
        with st.spinner("Running PyTorch PINN inference..."):
            # Generate spatial grid from rotor (0.05m) to stator (0.10m)
            radii = np.linspace(0.05, 0.10, 100).reshape(-1, 1)
            y_coords = np.zeros_like(radii)
            
            # Map parameters to network inputs
            t_array = np.full_like(radii, grinding_time * 60)
            rpm_array = np.full_like(radii, target_rpm)
            X_infer = np.hstack((radii, y_coords, t_array, rpm_array))
            
            # Forward pass inference
            predictions = model.predict(X_infer)
            predicted_D90 = predictions[:, 4]
            final_D90 = predicted_D90[-1]
            
            st.success("Simulation Complete")
            st.metric(label="Predicted Outlet D90", value=f"{final_D90:.2f} µm")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(radii * 100, predicted_D90, color="#0052cc", linewidth=2.5)
            ax.set_title(f"Internal D90 Profile at {int(target_rpm)} RPM after {int(grinding_time)} min")
            ax.set_xlabel("Radial Distance from Center (cm)")
            ax.set_ylabel("D90 Particle Size (µm)")
            ax.grid(True, linestyle="--", alpha=0.7)
            
            st.pyplot(fig)
