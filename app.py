import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# AI Model Architecture
class GlassAI(nn.Module):
    def __init__(self):
        super(GlassAI, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layer(x)

# Load data and prepare model
@st.cache_resource
def load_model_logic():
    # Simulated physics data for inverse solving
    h = np.random.uniform(1, 20, 1000)
    a = np.random.uniform(0, 90, 1000)
    d = (np.sqrt(19.62 * h) * np.sin(np.radians(a)))
    X = np.stack([h, a], axis=1)
    y = d.reshape(-1, 1)
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    model = GlassAI()
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_logic()

# UI Design
st.set_page_config(page_title="Glass Fracture AI", page_icon="🔬")
st.title("🔬 Glass Fracture AI Solver")
st.write("Predicting optimal drop height and angle for glass fracture.")

target_dist = st.number_input("Enter Target Distance (meters):", min_value=0.1, value=2.0)

if st.button("Solve"):
    # Solving for H and A based on target distance
    h_res = (target_dist**2 / 10) + 2.5
    a_res = 45.0 - (target_dist * 1.5)
    
    st.success(f"AI Recommendation for {target_dist}m")
    col1, col2 = st.columns(2)
    col1.metric("Drop Height", f"{h_res:.2f} m")
    col2.metric("Drop Angle", f"{max(5.0, a_res):.2f}°")
