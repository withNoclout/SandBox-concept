import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from soh_simulation import simulate_battery_life
from battery_monitor import get_battery_stats

st.set_page_config(page_title="Battery SOH Simulator", page_icon="🔋", layout="wide")

# --- SIDEBAR: REAL DATA ---
st.sidebar.header("🔋 Real-Time Battery Stats")
stats = get_battery_stats()

if stats:
    st.sidebar.metric("Current SOH", f"{stats['SOH']:.1f}%")
    st.sidebar.metric("Cycle Count", f"{stats['CycleCount']}")
    st.sidebar.metric("Max Capacity", f"{stats['MaxCapacity']} mAh")
    st.sidebar.metric("Design Capacity", f"{stats['DesignCapacity']} mAh")
    st.sidebar.metric("Temperature", f"{stats['Temperature']:.1f} °C")
    st.sidebar.metric("Voltage", f"{stats['Voltage']:.2f} V")
    st.sidebar.write(f"**Status:** {'Charging ⚡' if stats['IsCharging'] else 'Discharging 🔋'}")
    
    initial_soh = stats['SOH']
    initial_cycles = stats['CycleCount']
else:
    st.sidebar.error("Could not fetch battery stats. Using defaults.")
    initial_soh = 100.0
    initial_cycles = 0

# --- MAIN: SIMULATION CONTROLS ---
st.title("🔋 Battery Life Predictor")
st.markdown("Simulate how your battery will degrade based on usage patterns, starting from your **current** health.")

col1, col2, col3 = st.columns(3)

with col1:
    avg_temp = st.slider("Avg Temperature (°C)", 10, 60, int(stats['Temperature']) if stats else 25)
    
with col2:
    charge_limit = st.slider("Charge Limit (%)", 50, 100, 80)

with col3:
    c_rate = st.slider("Discharge Rate (C-rate)", 0.1, 2.0, 0.5)

dod = st.slider("Depth of Discharge (DoD)", 0.1, 1.0, 0.5, help="How much you use before recharging (1.0 = 100% to 0%)")

# --- RUN SIMULATION ---
# Scenario 1: User Configured
history_user = simulate_battery_life(
    avg_temp=avg_temp, 
    charge_limit=charge_limit, 
    dod=dod, 
    c_rate=c_rate,
    initial_soh=initial_soh,
    initial_cycles=initial_cycles
)

# Scenario 2: Optimized (Cool, 80% limit, Low DoD)
history_opt = simulate_battery_life(
    avg_temp=25, 
    charge_limit=80, 
    dod=0.4, 
    c_rate=0.5,
    initial_soh=initial_soh,
    initial_cycles=initial_cycles
)

# Scenario 3: Worst Case (Hot, 100% limit, High DoD)
history_worst = simulate_battery_life(
    avg_temp=45, 
    charge_limit=100, 
    dod=0.9, 
    c_rate=1.0,
    initial_soh=initial_soh,
    initial_cycles=initial_cycles
)

# --- PLOTTING ---
st.subheader("Degradation Projection")

fig, ax = plt.subplots(figsize=(10, 5))
cycles = np.arange(initial_cycles, initial_cycles + len(history_user))

ax.plot(cycles, history_user, label='Your Scenario', color='blue', linewidth=2)
ax.plot(cycles, history_opt, label='Optimized (Ideal)', color='green', linestyle='--')
ax.plot(cycles, history_worst, label='Worst Case (Abuse)', color='red', linestyle='--')

ax.axhline(y=80, color='grey', linestyle=':', label='Replacement (80%)')
ax.set_xlabel("Total Cycle Count")
ax.set_ylabel("State of Health (%)")
ax.set_title(f"Projected SOH starting from {initial_soh:.1f}%")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# --- ANALYSIS ---
st.subheader("Analysis")
end_soh = history_user[-1]
st.write(f"After **1000 more cycles**, your SOH is projected to be **{end_soh:.1f}%**.")

if end_soh < 80:
    st.warning("⚠️ Your battery will need replacement within this period!")
else:
    st.success("✅ Your battery will likely remain healthy.")
