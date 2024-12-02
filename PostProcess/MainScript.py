# %%

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# File paths
input_base_path = Path("2024_11_27")
output_directory = input_base_path / "PP"

# Load calibration files
torque_calib_file = input_base_path / "torque_calib_baseline.txt"
thrust_calib_file = input_base_path / "thrust_calib_baseline.txt"

torque_calib = pd.read_csv(torque_calib_file, delimiter='\t')
thrust_calib = pd.read_csv(thrust_calib_file, delimiter='\t')


# Read Torque calibration data
CalDataNm = pd.read_csv(f"{input_base_path}\\torque_calib_baseline.txt", delimiter='\t')

# Reference values for calibration and adjustments
refCentre1 = CalDataNm['LoadL'][1]
refCentre2 = CalDataNm['LoadR'][1]
CalDataNm['LoadL'] -= refCentre1
CalDataNm['LoadR'] -= refCentre2

# Calibration parameters
RefPoints = np.array([0, 0.05, 0.1, 0.2, -0.05, -0.1, -0.2] ) * 9.82 * 0.1
NmCalc = 0.019 * (CalDataNm['LoadL'] + CalDataNm['LoadR'])
CalPoints = NmCalc[[1 , 2 , 4 , 6 , 9 , 11, 13 ]]
torque_coeff = np.polyfit(CalPoints, RefPoints, 1)

# Read Thrust calibration data
CalDataT = pd.read_csv(f"{input_base_path}\\thrust_calib_baseline.txt", delimiter='\t')
RefPointsT = np.array([0,0.1 ,0.2,0.5,0.5,0.2,0.1,0]) * 9.82
TMeas = CalDataT['Thrust']
CalPointsT = TMeas[[0,1,3,5,7,9,11,12]]
thrust_coeff = np.polyfit(CalPointsT, RefPointsT, 1)

# Plot calibration data with subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Torque Calibration Plot
axs[0].plot(RefPoints, CalPoints, '+')
axs[0].plot(np.polyval(p, CalPoints), CalPoints, '-')
axs[0].set_xlabel('Reference Points (Torque)')
axs[0].set_ylabel('Calculated Points')
axs[0].legend(['Data Points', 'Linear Fit'])
axs[0].set_title('Torque Calibration')

# Thrust Calibration Plot
axs[1].plot(RefPointsT, CalPointsT, '+')
axs[1].plot(np.polyval(pT, CalPointsT), CalPointsT, '-')
axs[1].set_xlabel('Reference Points (Thrust)')
axs[1].set_ylabel('Calculated Points')
axs[1].legend(['Data Points', 'Linear Fit'])
axs[1].set_title('Thrust Calibration')

# Show the calibration figure
plt.tight_layout()
plt.show()


# %%
# Load test files
test_files = [
    "T1_Baseline_042_062_23ms.txt",
    "T2_Baseline_042_062_23ms_Rep1.txt",
    "T2_Baseline_042_062_23ms_Rep2.txt",
    "T3_Baseline_042_066_23ms_Rep2.txt",
    "T3_Baseline_042_066_23ms_Rep3.txt",
    "T3_Baseline_042_066_23ms_Rep4.txt",
    "T10_Severe_042_066_23ms_Rep1.txt",
    "T11_Severe_042_066_23ms_Rep1.txt",
    "T11_Severe_042_066_23ms_Rep2.txt",
    "T12_Severe_042_066_23ms_Rep1.txt"
]

# Define calibration functions
def calibrate_torque(loadL, loadR, coeff):
    return coeff[0] * (loadL + loadR) * 0.019 + coeff[1]

def calibrate_thrust(raw_thrust, coeff):
    return coeff[0] * raw_thrust + coeff[1]

# %% 

# Process test files
results = {}
for file_name in test_files:
    file_path = input_base_path / file_name
    test_data = pd.read_csv(file_path, delimiter='\t')
    
    # Apply calibrations
    test_data["Torque"] = calibrate_torque(test_data["LoadL"], test_data["LoadR"], torque_coeff)
    test_data["Thrust"] = calibrate_thrust(test_data["RawThr"], thrust_coeff)
    
    # Calculate derived metrics
    dia = 0.276  # Propeller diameter in meters
    test_data["n"] = test_data["RPM"] / 60  # Revolutions per second
    test_data["J"] = test_data["U"] / (test_data["n"] * dia)
    test_data["Ct"] = test_data["Thrust"] / (test_data["rho"] * (test_data["n"] ** 2) * (dia ** 4))
    test_data["P"] = 2 * np.pi * test_data["n"] * test_data["Torque"]
    test_data["Cp"] = test_data["P"] / (test_data["rho"] * (test_data["n"] ** 3) * (dia ** 5))
    test_data["eta"] = test_data["J"] * test_data["Ct"] / test_data["Cp"]
    
    # Store results
    results[file_name] = test_data[["J", "Ct", "Cp", "eta"]]
    # Save processed data
    test_data[["J", "Ct", "Cp", "eta"]].to_csv(output_directory / f"{file_name}_processed.txt", index=False, sep="\t")

# Statistical analysis for consistency
baseline_data = pd.concat([results[file] for file in test_files if "Baseline" in file])
severe_data = pd.concat([results[file] for file in test_files if "Severe" in file])

# Perform t-tests on metrics (J, Ct, Cp, eta)
metrics = ["Ct", "Cp", "eta"]
t_test_results = {metric: ttest_ind(baseline_data[metric], severe_data[metric]) for metric in metrics}

# Visualize comparison
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.boxplot([baseline_data[metric], severe_data[metric]], labels=["Baseline", "Severe"])
    plt.title(f"Comparison of {metric} between Baseline and Severe Cases")
    plt.ylabel(metric)
    plt.show()

# Output t-test results
t_test_results

# %%
# Plot processed data for all cases
plt.figure(figsize=(15, 10))

# Plot eta
plt.subplot(3, 1, 1)
for file_name, data in results.items():
    filtered_data = data[(data["J"] >= 0.6) & (data["J"] <= 0.9)]
    plt.scatter(filtered_data["J"], filtered_data["eta"], label=file_name)
plt.title("Propeller Efficiency (eta) vs Advance Ratio (J)")
plt.xlabel("Advance Ratio (J)")
plt.ylabel("Efficiency (eta)")
plt.legend()

# Plot Cp
plt.subplot(3, 1, 2)
for file_name, data in results.items():
    filtered_data = data[(data["J"] >= 0.6) & (data["J"] <= 0.9)]
    plt.scatter(filtered_data["J"], filtered_data["Cp"], label=file_name)
plt.title("Power Coefficient (Cp) vs Advance Ratio (J)")
plt.xlabel("Advance Ratio (J)")
plt.ylabel("Power Coefficient (Cp)")
plt.legend()

# Plot Ct
plt.subplot(3, 1, 3)
for file_name, data in results.items():
    filtered_data = data[(data["J"] >= 0.6) & (data["J"] <= 0.9)]
    plt.scatter(filtered_data["J"], filtered_data["Ct"], label=file_name)
plt.title("Thrust Coefficient (Ct) vs Advance Ratio (J)")
plt.xlabel("Advance Ratio (J)")
plt.ylabel("Thrust Coefficient (Ct)")
plt.legend()

plt.tight_layout()
plt.show()
# %%
