import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION (ค่าพารามิเตอร์เริ่มต้น) ---
Initial_Capacity = 100  # SOH เริ่มต้น 100%
Cycles_to_Simulate = 1000 # จำลองไปข้างหน้า 1000 รอบ (ประมาณ 3 ปี)

# --- 2. DEGRADATION LOGIC (สมการจำลองความเสื่อม) ---
def simulate_battery_life(avg_temp, charge_limit, dod=0.5, c_rate=0.5, noise_level=0.01, initial_soh=100.0, initial_cycles=0):
    """
    Simulate battery SOH degradation over time.
    
    Parameters:
    - avg_temp: Average operating temperature (Celsius)
    - charge_limit: Max charge percentage (e.g., 80 or 100)
    - dod: Depth of Discharge (0.0 to 1.0)
    - c_rate: Discharge rate (e.g., 0.5C, 1.0C)
    - noise_level: Magnitude of random daily variance
    - initial_soh: Starting State of Health (%)
    - initial_cycles: Number of cycles already passed
    """
    soh_history = [initial_soh]
    current_soh = initial_soh
    
    # --- STRESS FACTORS ---
    # 1. Temperature Stress: Arrhenius-like effect (simplified)
    # Higher temp = faster chemical reaction = faster degradation
    temp_stress = 1 + max(0, (avg_temp - 25) * 0.05) 
    
    # 2. Voltage Stress: High SoC keeps voltage high, stressing cathode
    voltage_stress = 1.5 if charge_limit > 80 else 1.0 
    
    # 3. DoD Stress: Deeper cycles cause more mechanical stress on electrodes
    # Non-linear: 80% DoD is much worse than 2x 40% DoD
    dod_stress = (dod / 0.5) ** 1.5
    
    # 4. C-rate Stress: High current generates internal heat and lithium plating risk
    c_rate_stress = c_rate ** 1.2
    
    base_degradation = 0.02 

    # --- SIMULATION LOOP ---
    for cycle in range(Cycles_to_Simulate):
        # Calculate daily damage
        # Deterministic part
        damage = base_degradation * temp_stress * voltage_stress * dod_stress * c_rate_stress
        
        # Stochastic part (random usage fluctuations)
        noise = np.random.normal(0, noise_level * base_degradation)
        damage = max(0, damage + noise) # Damage can't be negative
        
        # Update SOH
        current_soh -= damage
        soh_history.append(current_soh)
        
    return soh_history

if __name__ == "__main__":
    scenario_unmanaged = simulate_battery_life(avg_temp=35, charge_limit=100)
    scenario_managed = simulate_battery_life(avg_temp=28, charge_limit=80)
    plt.figure(figsize=(10, 6))
    plt.plot(scenario_unmanaged, label='Unmanaged (High Temp, 100% Charge)', color='red', linestyle='--')
    plt.plot(scenario_managed, label='Smart Maintenance (Cool, 80% Limit)', color='green')

    # เส้นขีดจำกัดเสื่อม (End of Life) ปกติคือ 80%
    plt.axhline(y=80, color='grey', linestyle=':', label='Replacement Threshold (80%)')

    plt.title('Battery Life Simulation: Predictive Maintenance Project')
    plt.xlabel('Charge Cycles (Approx. Days)')
    plt.ylabel('State of Health (SOH) %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
