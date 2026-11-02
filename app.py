import streamlit as st
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time

st.set_page_config(
    page_title="Bead Mill Digital Twin | Process Engineering", 
    layout="wide"
)

@st.cache_resource
def load_digital_twin():
    """Rebuilds the architecture and loads the trained PyTorch weights natively."""
    net = dde.nn.FNN([4] + [40] * 3 + [5], "tanh", "Glorot normal")
    model = dde.Model(data=None, net=net)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(BASE_DIR, "models", "beadmill_pinn_weights.pt")
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.net.load_state_dict(state_dict)
        return model
    else:
        return None

model = load_digital_twin()

st.title("Bead Mill Digital Twin: Multiphysics & Energy Predictor")
st.markdown("Physics-Informed Neural Network (PINN) inference integrating Kwade's Stress Model.")

with st.sidebar:
    st.header("Process Parameters")
    target_rpm = st.slider("Agitator Speed (RPM)", 500.0, 3000.0, 1500.0, step=100.0)
    grinding_time = st.slider("Grinding Time (minutes)", 0.0, 120.0, 60.0, step=5.0)
    bead_density = st.number_input("Bead Density (kg/m³)", value=6000) # Default YSZ beads
    bead_diam_mm = st.number_input("Bead Diameter (mm)", value=0.5, step=0.1)
    
    run_sim = st.button("Simulate Batch", type="primary", use_container_width=True)

if run_sim:
    if model is None:
        st.error("Model weights not found. Please ensure 'beadmill_pinn_weights.pt' is in the 'models/' directory.")
    else:
        # 1. Setup Grid & Tensor Inference
        radii = np.linspace(0.05, 0.10, 100).reshape(-1, 1) # Rotor to Stator
        y_coords = np.zeros_like(radii)
        t_array = np.full_like(radii, grinding_time * 60)
        rpm_array = np.full_like(radii, target_rpm)
        
        X_infer = np.hstack((radii, y_coords, t_array, rpm_array))
        X_tensor = torch.tensor(X_infer, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model.net(X_tensor).cpu().numpy()
            
        u, v, p, phi, d90 = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4]
        
        # 2. Physics & Energy Calculations (Kwade's Model)
        # Velocity and Shear
        v_mag = np.sqrt(u**2 + v**2)
        dr = radii[1, 0] - radii[0, 0]
        shear_rate = np.abs(np.gradient(v_mag, dr)) # dv/dr
        
        # Stress Energy (SE) = 1/2 * m_bead * v_tip^2
        v_tip = (target_rpm * 2 * np.pi / 60) * 0.05 # m/s at rotor tip
        bead_vol = (np.pi / 6) * ((bead_diam_mm / 1000)**3)
        m_bead = bead_vol * bead_density
        stress_energy = 0.5 * m_bead * (v_tip**2)
        
        # Stress Number (SN) proportional to RPM, time, and avg bead concentration
        stress_number = target_rpm * grinding_time * np.mean(phi) * 100 
        
        # Specific Energy (approximate particle reduction energy)
        specific_energy = stress_energy * stress_number
        
        # 3. Dashboard Metrics
        st.subheader("Process Energetics & Final State")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Outlet D90", f"{d90[-1]:.2f} µm")
        col2.metric("Stress Energy (SE)", f"{stress_energy * 1e6:.2f} µJ")
        col3.metric("Stress Number (SN)", f"{stress_number:,.0f}")
        col4.metric("Total Specific Energy", f"{specific_energy * 1e3:.2f} mJ")

        # 4. Interactive Tabs for Profiles
        tab1, tab2, tab3, tab4 = st.tabs(["Particle Size", "Velocity & Shear", "Bead Concentration", "Time Animation"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(radii * 100, d90, color="#0052cc", linewidth=2.5)
            ax.set_title("Internal D90 Profile at End of Batch")
            ax.set_xlabel("Radial Distance from Center (cm)")
            ax.set_ylabel("D90 Particle Size (µm)")
            ax.grid(True, linestyle="--", alpha=0.7)
            st.pyplot(fig)
            
        with tab2:
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax2 = ax1.twinx()
            
            ax1.plot(radii * 100, v_mag, color="green", linewidth=2.5, label="Velocity")
            ax2.plot(radii * 100, shear_rate, color="red", linewidth=2.5, linestyle=":", label="Shear Rate")
            
            ax1.set_xlabel("Radial Distance from Center (cm)")
            ax1.set_ylabel("Velocity Magnitude (m/s)", color="green")
            ax2.set_ylabel("Shear Rate (1/s)", color="red")
            ax1.set_title("Hydrodynamic Flow Profile")
            ax1.grid(True, linestyle="--", alpha=0.7)
            
            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
            st.pyplot(fig)
            
        with tab3:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(radii * 100, phi, color="orange", linewidth=2.5)
            ax.set_title("Bead Volume Fraction Distribution")
            ax.set_xlabel("Radial Distance from Center (cm)")
            ax.set_ylabel("Bead Concentration (φ)")
            ax.set_ylim(0, 0.7) # Max packing is typically 0.65
            ax.grid(True, linestyle="--", alpha=0.7)
            st.pyplot(fig)
            
        with tab4:
            st.markdown("**D90 Reduction Over Time**")
            plot_spot = st.empty()
            progress_bar = st.progress(0)
            
            # Generate 20 time steps for the animation
            time_steps = np.linspace(0, grinding_time * 60, 20)
            
            for i, t_step in enumerate(time_steps):
                X_t = np.hstack((radii, y_coords, np.full_like(radii, t_step), rpm_array))
                with torch.no_grad():
                    d90_t = model.net(torch.tensor(X_t, dtype=torch.float32)).cpu().numpy()[:, 4]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(radii * 100, d90_t, color="#0052cc", linewidth=2.5)
                ax.set_title(f"D90 Profile at {t_step/60:.1f} minutes")
                ax.set_xlabel("Radial Distance from Center (cm)")
                ax.set_ylabel("D90 Particle Size (µm)")
                ax.set_ylim(0, 160) # Lock y-axis so the drop is visually apparent
                ax.grid(True, linestyle="--", alpha=0.7)
                
                plot_spot.pyplot(fig)
                plt.close(fig) # Prevent memory leaks
                progress_bar.progress((i + 1) / len(time_steps))
                time.sleep(0.1) # Brief pause for visual effect
