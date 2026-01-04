import subprocess
import re

def get_battery_stats():
    """
    Fetches battery statistics from macOS using `ioreg`.
    Returns a dictionary with keys:
    - CycleCount (int)
    - MaxCapacity (int)
    - DesignCapacity (int)
    - Temperature (float, Celsius)
    - Voltage (float, Volts)
    - IsCharging (bool)
    - SOH (float, %)
    """
    try:
        # Run ioreg command
        result = subprocess.run(
            ["ioreg", "-r", "-n", "AppleSmartBattery", "-w0"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Parse fields using regex
        stats = {}
        
        # Helper to extract int values
        def extract_int(key):
            match = re.search(f'"{key}" = (\d+)', output)
            return int(match.group(1)) if match else None

        # Helper to extract string/bool
        def extract_str(key):
            match = re.search(f'"{key}" = (.+)', output)
            return match.group(1).strip() if match else None

        stats['CycleCount'] = extract_int("CycleCount")
        stats['MaxCapacity'] = extract_int("AppleRawMaxCapacity") # Using Raw for accuracy
        stats['DesignCapacity'] = extract_int("DesignCapacity")
        
        # Temperature is usually in 0.01 Kelvin or Celsius. 
        # Based on typical values (3000-3500), it's likely 0.01 C (30.00 C).
        # Let's assume 0.01 C for now based on common macOS behavior.
        temp_raw = extract_int("Temperature")
        stats['Temperature'] = temp_raw / 100.0 if temp_raw else 0.0
        
        # Voltage is in mV
        volt_raw = extract_int("Voltage")
        stats['Voltage'] = volt_raw / 1000.0 if volt_raw else 0.0
        
        charging_str = extract_str("IsCharging")
        stats['IsCharging'] = True if charging_str == "Yes" else False
        
        # Calculate SOH
        if stats['MaxCapacity'] and stats['DesignCapacity']:
            stats['SOH'] = (stats['MaxCapacity'] / stats['DesignCapacity']) * 100.0
        else:
            stats['SOH'] = 0.0
            
        return stats

    except Exception as e:
        print(f"Error fetching battery stats: {e}")
        return None

if __name__ == "__main__":
    data = get_battery_stats()
    if data:
        print("🔋 Battery Stats:")
        for k, v in data.items():
            print(f"  {k}: {v}")
    else:
        print("❌ Failed to get data.")
