# %% Kalibrering
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sympy import symbols
from Taylor_error import Taylor_error_propagation

# Close any open plots
plt.close('all')

# Set base paths for input and output files
input_base_path = r"2024_11_27"
output_directory = Path(r"2024_11_27\PP")

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
p = np.polyfit(CalPoints, RefPoints, 1)

# Read Thrust calibration data
CalDataT = pd.read_csv(f"{input_base_path}\\thrust_calib_baseline.txt", delimiter='\t')
RefPointsT = np.array([0,0.1 ,0.2,0.5,0.5,0.2,0.1,0]) * 9.82
TMeas = CalDataT['Thrust']
CalPointsT = TMeas[[0,1,3,5,7,9,11,12]]
pT = np.polyfit(CalPointsT, RefPointsT, 1)

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


# %% Råda plot med kaliberering

input_base_path = Path(input_base_path)

# List all test files, excluding specific ones
exclude_files = {'thrust_calib_baseline.txt', 'torque_calib_baseline.txt'}
test_files = sorted(
    file for file in input_base_path.glob("T*_*.txt") if file.name not in exclude_files
)


#%%

# Function to process test data
def process_test_data(file_path, p_torque, p_thrust):
    test_data = pd.read_csv(file_path, delimiter='\t')

    dia = 0.276 # Diameter of propeller in meters

    # Calculate derived metrics
    test_data['torque'] = abs(0.019 * (test_data['LoadL'] + test_data['LoadR']) * p_torque[0])
    test_data['n'] = test_data['RPM'] / 60  # Revolutions per second
    rho = test_data['rho']
    test_data['J'] = test_data['U'] / (test_data['n'] * dia)
    test_data['Thrust'] *= p_thrust[0]
    test_data['Ct'] = test_data['Thrust'] / (rho * (test_data['n'] ** 2) * (dia ** 4))
    test_data['P'] = 2 * np.pi * test_data['n'] * test_data['torque']
    test_data['Cp'] = test_data['P'] / (rho * (test_data['n'] ** 3) * (dia ** 5))
    test_data['eta'] = test_data['J'] * test_data['Ct'] / test_data['Cp']

    return test_data

# Prepare lists to accumulate data for plotting
all_ct = []
all_cp = []
all_eta = []
all_labels = []

datasets = {}

# Process each test file and store data for cumulative plotting
for file in test_files:
    try:
        test_data = process_test_data(file, p, pT)
        datasets[file.stem] = test_data
        
        # Accumulate data
        all_ct.append((test_data['J'], test_data['Ct']))
        all_cp.append((test_data['J'], test_data['Cp']))
        all_eta.append((test_data['J'], test_data['eta']))
        all_labels.append(file.stem)

        # Save trimmed results
        result = test_data[['J', 'Ct', 'Cp', 'eta']]
        result.to_csv(output_directory / f"{file.stem}_pp.txt", sep=';', index=False)
    
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Plot Ct and Cp in the same figure
plt.figure(figsize=(10, 6))
for (j_vals, ct_vals), label in zip(all_ct, all_labels):
    plt.scatter(j_vals, ct_vals, label=f'{label} - Ct', marker='o')
for (j_vals, cp_vals), label in zip(all_cp, all_labels):
    plt.scatter(j_vals, cp_vals, label=f'{label} - Cp', marker='+')
plt.xlabel('Advance Ratio (J)')
plt.ylabel('Ct / Cp')
plt.title('Comparison of Ct and Cp Across Datasets')
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.xlim(0.6,0.85)
plt.ylim(0,0.04)
plt.show()

# Plot Efficiency (Eta) in a separate figure
plt.figure(figsize=(10, 6))
for (j_vals, eta_vals), label in zip(all_eta, all_labels):
    plt.scatter(j_vals, eta_vals, label=label, marker='+')
plt.xlabel('Advance Ratio (J)')
plt.ylabel('Efficiency (Eta)')
plt.title('Comparison of Efficiency (Eta) Across Datasets')
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.xlim(0.6,0.85)
plt.ylim(0,1)
plt.show()


# %%

baseline_keys = ['T1_Baseline_042_062_23ms', 'T2_Baseline_042_062_23ms_Rep1', 'T2_Baseline_042_062_23ms_Rep2', 'T3_Baseline_042_066_23ms_Rep2', 'T3_Baseline_042_066_23ms_Rep3', 'T3_Baseline_042_066_23ms_Rep4']
baseline_data = {key: datasets[key] for key in baseline_keys}

# Gather thrust data from all baseline datasets, taking the sample range from 2 to end
thrust_data = pd.DataFrame({key: data['Thrust'].iloc[2:-1].reset_index(drop=True) for key, data in baseline_data.items()})

# Compute mean and standard deviation of thrust
thrust_mean = thrust_data.mean(axis=1)
thrust_std = thrust_data.std(axis=1)



# %%
# Define symbols
T, rho, n, d, Q, omega, V = symbols('T rho n d Q omega V')

# Define expressions for CT, CP, J, and eta
CT_Expr = T / (rho * n**2 * d**4)
CP_Expr = (Q * omega) / (rho * n**3 * d**5)
J_Expr = V / (n * d)
Eta_Expr = (CT_Expr * J_Expr) / CP_Expr


# Extract variables from the first dataset in datasets
first_dataset_key = next(iter(datasets))
TestData = datasets[first_dataset_key]

# Extract variables from the data
T_values = TestData["Thrust"].iloc[2:-1].reset_index(drop=True)  # Thrust (N)
Q_values = TestData["Torque"].iloc[2:-1].reset_index(drop=True)  # Torque (Nm)
RPS_values = TestData["n"].iloc[2:-1].reset_index(drop=True)     # Rotational speed 
rho_values = TestData["rho"].iloc[2:-1].reset_index(drop=True)   # Air density (kg/m^3)
CT_values = TestData['Ct'].iloc[2:-1].reset_index(drop=True)
CP_values = TestData['Cp'].iloc[2:-1].reset_index(drop=True)
Eta_values = TestData['eta'].iloc[2:-1].reset_index(drop=True)
J_num = TestData['J'].iloc[2:-1].reset_index(drop=True)
d_value = 0.276                # Propeller diameter (m), assumed constant
V_value = 23                   # Air velocity (m/s), assumed constant

# Convert RPM to n (revolutions per second) and omega (angular velocity)
omega_values = 2 * np.pi * RPS_values  # Angular velocity (rad/s)

# Placeholder for computed values and errors
CT_errors = []
CP_errors = []
Eta_errors = []

# Thrust and torque sensor uncertainties
full_scale_thrust = 9.81  # Full scale in N (1 kg capacity = 9.81 N)
thrust_uncertainty = np.sqrt(
    (0.05 / 100 * full_scale_thrust)**2 +  # Non-linearity
    (0.05 / 100 * full_scale_thrust)**2 +  # Hysteresis
    (0.03 / 100 * full_scale_thrust)**2 +  # Repeatability
    (1 / 100 * full_scale_thrust)**2 +     # Zero balance
    (0.05 / 100 * full_scale_thrust)**2    # Creep
)

full_scale_torque = 1 * 9.81 * 0.019  # 1 kg capacity at 19 mm arm = 0.18639 Nm
torque_uncertainty = np.sqrt(
    (0.05 / 100 * full_scale_torque)**2 +  # Non-linearity
    (0.03 / 100 * full_scale_torque)**2 +  # Reproducibility
    (0.03 / 100 * full_scale_torque)**2 +  # Hysteresis
    (0.1 / 100 * full_scale_torque)**2 +   # Zero-point comparison
    (0.1 / 100 * full_scale_torque)**2     # Creep
)





# Sensor uncertainties
uncertainties = {
    'T': thrust_std,  # Thrust uncertainty (N)
    'rho': 0,                 # Air density uncertainty
    'n': 10 / 60,             # RPM uncertainty
    'd': 0.000049,             # Diameter uncertainty prusa SLS 0.049 Printer Precision (m)
    'Q': torque_uncertainty,  # Torque uncertainty (Nm)
    'omega': 0,               # Angular velocity uncertainty (rad/s)
    'V': 0.1                  # Air velocity uncertainty (m/s)
}



# Loop through data points to compute CT, CP, Eta, and uncertainties

for i in range(len(T_values)):
    T_value = T_values[i]  
    Q_value = Q_values[i]
    n_value = RPS_values[i]
    rho_value = rho_values[i]
    omega_value = omega_values[i]

    # Define variable inputs for the Taylor propagation
    dict_variables_input = {
        'Variables': [T, rho, n, d, Q, omega, V],
        'Values': [T_value, rho_value, n_value, d_value, Q_value, omega_value, V_value],
        'Error_type': ['abs', 'abs', 'abs', 'abs', 'abs', 'abs', 'abs'],
        'Error': [uncertainties['T'][i], uncertainties['rho'], uncertainties['n'],
                  uncertainties['d'], uncertainties['Q'], uncertainties['omega'], uncertainties['V']]
    }

    # Compute uncertainties
    CT_uncertainties, CT_sum_uncert, _, _ = Taylor_error_propagation(CT_Expr, dict_variables_input)
    CP_uncertainties, CP_sum_uncert, _, _ = Taylor_error_propagation(CP_Expr, dict_variables_input)
    Eta_uncertainties, Eta_sum_uncert, _, _ = Taylor_error_propagation(Eta_Expr, dict_variables_input)

    # Append results
    CT_errors.append(CT_sum_uncert)
    CP_errors.append(CP_sum_uncert)
    Eta_errors.append(Eta_sum_uncert)


# Plot CT, CP, and Eta with shaded error areas
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

CT_values = np.array(CT_values, dtype=np.float64)
CT_errors = np.array(CT_errors, dtype=np.float64)
CP_values = np.array(CP_values, dtype=np.float64)
CP_errors = np.array(CP_errors, dtype=np.float64)
Eta_values = np.array(Eta_values, dtype=np.float64)
Eta_errors = np.array(Eta_errors, dtype=np.float64)
J_num = np.array(J_num, dtype=np.float64)

# Plot CT
axs[0].fill_between(J_num, CT_values - CT_errors, CT_values + CT_errors, color='grey', alpha=0.3)
axs[0].scatter(J_num, CT_values, color='blue', label='Sample Points')
axs[0].set_title('Thrust Coefficient (C_T) vs J', fontsize=14)
axs[0].set_xlabel('J', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].legend()
axs[0].set_xlim(0.5, 0.95)

# Plot CP
axs[1].scatter(J_num, CP_values, color='blue', label='Sample Points')
axs[1].fill_between(J_num, CP_values - CP_errors, CP_values + CP_errors, color='grey', alpha=0.3)
axs[1].set_title('Power Coefficient (C_P) vs J', fontsize=14)
axs[1].set_xlabel('J', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].legend()
axs[1].set_xlim(0.5, 0.95)

# Plot Eta
axs[2].scatter(J_num, Eta_values, color='blue', label='Sample Points')
axs[2].fill_between(J_num, Eta_values - Eta_errors, Eta_values + Eta_errors, color='grey', alpha=0.3)
axs[2].set_title('Efficiency (Eta) vs J', fontsize=14)
axs[2].set_xlabel('J', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].grid(True)
axs[2].legend()
axs[2].set_xlim(0.5, 0.95)

plt.tight_layout()
plt.show()

# %%

# Prepare lists to accumulate data for plotting with errors
all_ct_errors = []
all_cp_errors = []
all_eta_errors = []

# Loop through each baseline dataset to compute errors
for key in baseline_keys:
    TestData = datasets[key]

    # Extract variables from the data
    T_values = TestData["Thrust"].iloc[2:-1].reset_index(drop=True)  # Thrust (N)
    Q_values = TestData["Torque"].iloc[2:-1].reset_index(drop=True)  # Torque (Nm)
    RPS_values = TestData["n"].iloc[2:-1].reset_index(drop=True)     # Rotational speed 
    rho_values = TestData["rho"].iloc[2:-1].reset_index(drop=True)   # Air density (kg/m^3)
    CT_values = TestData['Ct'].iloc[2:-1].reset_index(drop=True)
    CP_values = TestData['Cp'].iloc[2:-1].reset_index(drop=True)
    Eta_values = TestData['eta'].iloc[2:-1].reset_index(drop=True)
    J_num = TestData['J'].iloc[2:-1].reset_index(drop=True)
    omega_values = 2 * np.pi * RPS_values  # Angular velocity (rad/s)

    # Placeholder for computed values and errors
    CT_errors = []
    CP_errors = []
    Eta_errors = []

    # Loop through data points to compute CT, CP, Eta, and uncertainties
    for i in range(len(T_values)):
        T_value = T_values[i]  
        Q_value = Q_values[i]
        n_value = RPS_values[i]
        rho_value = rho_values[i]
        omega_value = omega_values[i]

        # Define variable inputs for the Taylor propagation
        dict_variables_input = {
            'Variables': [T, rho, n, d, Q, omega, V],
            'Values': [T_value, rho_value, n_value, d_value, Q_value, omega_value, V_value],
            'Error_type': ['abs', 'abs', 'abs', 'abs', 'abs', 'abs', 'abs'],
            'Error': [uncertainties['T'][i], uncertainties['rho'], uncertainties['n'],
                      uncertainties['d'], uncertainties['Q'], uncertainties['omega'], uncertainties['V']]
        }

        # Compute uncertainties
        CT_uncertainties, CT_sum_uncert, _, _ = Taylor_error_propagation(CT_Expr, dict_variables_input)
        CP_uncertainties, CP_sum_uncert, _, _ = Taylor_error_propagation(CP_Expr, dict_variables_input)
        Eta_uncertainties, Eta_sum_uncert, _, _ = Taylor_error_propagation(Eta_Expr, dict_variables_input)

        # Append results
        CT_errors.append(CT_sum_uncert)
        CP_errors.append(CP_sum_uncert)
        Eta_errors.append(Eta_sum_uncert)

    # Accumulate data
    all_ct.append((J_num, CT_values))
    all_cp.append((J_num, CP_values))
    all_eta.append((J_num, Eta_values))
    all_ct_errors.append(CT_errors)
    all_cp_errors.append(CP_errors)
    all_eta_errors.append(Eta_errors)
    all_labels.append(key)

    print(all_ct_errors)
    print(all_ct)

# %%
# Plot CT, CP, and Eta with shaded error areas for all datasets
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot CT
for (j_vals, ct_vals), ct_errs, label in zip(all_ct, all_ct_errors, all_labels):
    axs[0].fill_between(j_vals, ct_vals - ct_errs, ct_vals + ct_errs, color='grey', alpha=0.3)
    axs[0].scatter(j_vals, ct_vals, label=label, marker='o')
axs[0].set_title('Thrust Coefficient (C_T) vs J', fontsize=14)
axs[0].set_xlabel('J', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].legend()
axs[0].set_xlim(0.5, 0.95)

# Plot CP
for (j_vals, cp_vals), cp_errs, label in zip(all_cp, all_cp_errors, all_labels):
    axs[1].fill_between(j_vals, cp_vals - cp_errs, cp_vals + cp_errs, color='grey', alpha=0.3)
    axs[1].scatter(j_vals, cp_vals, label=label, marker='+')
axs[1].set_title('Power Coefficient (C_P) vs J', fontsize=14)
axs[1].set_xlabel('J', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].legend()
axs[1].set_xlim(0.5, 0.95)

# Plot Eta
for (j_vals, eta_vals), eta_errs, label in zip(all_eta, all_eta_errors, all_labels):
    axs[2].fill_between(j_vals, eta_vals - eta_errs, eta_vals + eta_errs, color='grey', alpha=0.3)
    axs[2].scatter(j_vals, eta_vals, label=label, marker='+')
axs[2].set_title('Efficiency (Eta) vs J', fontsize=14)
axs[2].set_xlabel('J', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].grid(True)
axs[2].legend()
axs[2].set_xlim(0.5, 0.95)

plt.tight_layout()
plt.show()
# %%
