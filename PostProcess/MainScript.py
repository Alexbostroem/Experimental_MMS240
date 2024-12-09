# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d
from Taylor_error import Taylor_error_propagation
from MCSolver_GPT import MonteCarlo_error_propagation
from sympy import symbols


# Read in CFD data of baseline to a pandas dataframe
df_baseline= pd.read_csv('../CFD/cfd_data_baseline_sweep.csv')
df_baseline.head()

df_baseline.columns = [
    "RPM", "Cp", "RPM_eta", "Eta", 
    "RPM_thrust", "Thrust", 
    "RPM_power", "Power", 
    "RPM_ct", "Ct", "RPM_J", "J", 
    "RPM_torque", "Torque"
]
df_baseline = df_baseline.drop(columns=[col for col in df_baseline.columns if col.startswith('RPM_')])


# Read in CFD data of severe to a pandas dataframe
df_severe= pd.read_csv('../CFD/cfd_data_severe_sweep.csv')
df_severe.head()

df_severe.columns = [
    "RPM", "Cp", "RPM_eta", "Eta", 
    "RPM_thrust", "Thrust", 
    "RPM_power", "Power", 
    "RPM_ct", "Ct", "RPM_J", "J", 
    "RPM_torque", "Torque"
]
df_severe = df_severe.drop(columns=[col for col in df_severe.columns if col.startswith('RPM_')])

# Close any open plots
plt.close('all')

# %% 
# Set base paths for input and output files
current_dir = os.path.dirname(os.path.abspath(__file__))
input_base_path = os.path.join(current_dir, "2024_11_27")
output_directory = os.path.join(current_dir, "2024_11_27/PP")

# Read Torque calibration data
CalDataNm = pd.read_csv(f"{input_base_path}/torque_calib_baseline.txt", delimiter='\t')

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
CalDataT = pd.read_csv(f"{input_base_path}/thrust_calib_baseline.txt", delimiter='\t')
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

# %%

filepath = f"{input_base_path}/T12_Severe_042_066_23ms_Rep1.txt"

# Read the test data
TestData = pd.read_csv(filepath, delimiter='\t')

# Calculate torque and other derived metrics
TestData['torque'] = abs(0.019 * (TestData['LoadL'] + TestData['LoadR']) * p[0])
Dia = 0.2286
TestData['n'] = TestData['RPM'] / 60  # Revolutions per second
rho = TestData['rho']
TestData['J'] = TestData['U'] / (TestData['n'] * Dia)
TestData['Thrust'] = TestData['Thrust'] * pT[0]
TestData['Ct'] = TestData['Thrust'] / (rho * (TestData['n'] ** 2) * (Dia ** 4))
TestData['P'] = 2 * np.pi * TestData['n'] * TestData['torque']
TestData['Cp'] = TestData['P'] / (rho * (TestData['n'] ** 3) * (Dia ** 5))
TestData['eta'] = TestData['J'] * TestData['Ct'] / TestData['Cp']



#%%


# Define symbols
T, rho, n, d, Q, omega, V = symbols('T rho n d Q omega V')

# Define expressions for CT, CP, J, and eta
CT_Expr = T / (rho * n**2 * d**4)
CP_Expr = (Q * omega) / (rho * n**3 * d**5)
J_Expr = V / (n * d)
Eta_Expr = (CT_Expr * J_Expr) / CP_Expr

# %%
# Placeholder for computed values and errors
CT_errors = []
CP_errors = []
Eta_errors = []

# Compute errors for baseline data

# Thrust and torque sensor uncertainties
full_scale_thrust = 9.81  # Full scale in N (1 kg capacity = 9.81 N)
thrust_uncertainty = np.sqrt(
    (0.05 / 100 * full_scale_thrust)**2   # Non-linearity

)

full_scale_torque = 1 * 9.81 * 0.019  # 1 kg capacity at 19 mm arm 
torque_uncertainty = np.sqrt(
    (0.05 / 100 * full_scale_torque)**2   # Non-linearity
)

for i in range(len(TestData['J'])):
    # Sensor uncertainties
    uncertainties = {
        'T': thrust_uncertainty,  # Thrust uncertainty (N) std_thrust_baseline[i]
        'rho': 0.0037,            # Air density uncertainty
        'n': 10/60 ,              # RPS uncertainty std_rpm_baseline[i]/60
        'd': 0.0003,              # Diameter uncertainty (m)
        'Q': torque_uncertainty,  # Torque uncertainty (Nm)
        'omega': 0 ,               # Angular velocity uncertainty (rad/s)
        'V': 0.1                  # Air velocity uncertainty (m/s)
    }

    T_value = TestData['Thrust'][i]
    Q_value = TestData['Torque'][i]
    n_value = TestData['n'][i]
    rho_value = 1.188
    omega_value = 2 * np.pi * n_value

    # Define variable inputs for the Taylor propagation
    dict_variables_input = {
        'Variables': [T, rho, n, d, Q, omega, V],
        'Values': [T_value, rho_value, n_value, Dia, Q_value, omega_value, 23],
        'Error_type': ['abs', 'abs', 'abs', 'abs', 'abs', 'abs', 'abs'],
        'Error': [uncertainties['T'], uncertainties['rho'], uncertainties['n'],
                  uncertainties['d'], uncertainties['Q'], uncertainties['omega'], uncertainties['V']]
    }

    # Compute uncertainties
    CT_uncertainties, CT_sum_uncert,_,_ = Taylor_error_propagation(CT_Expr, dict_variables_input)
    CP_uncertainties, CP_sum_uncert,_,_ = Taylor_error_propagation(CP_Expr, dict_variables_input)
    Eta_uncertainties, Eta_sum_uncert,_,_ = Taylor_error_propagation(Eta_Expr, dict_variables_input)

    
    # Append results
    CT_errors.append(CT_sum_uncert)
    CP_errors.append(CP_sum_uncert)
    Eta_errors.append(Eta_sum_uncert)

# %%
# Prepare data to save, trimming the first and last rows
A = TestData[['J', 'Ct', 'Cp', 'eta']].iloc[1:-1].to_numpy()
CT_errors_trimmed = np.array(CT_errors[1:-1])
CP_errors_trimmed = np.array(CP_errors[1:-1])
Eta_errors_trimmed = np.array(Eta_errors[1:-1])

# Combine data and errors
A_with_errors = np.column_stack((A, CT_errors_trimmed, CP_errors_trimmed, Eta_errors_trimmed))

# Generate output file path with `_pp.txt` suffix
output_file_path = os.path.join(output_directory, os.path.basename(filepath).replace('.txt', '_pp.txt'))

# Save data to file
np.savetxt(output_file_path, A_with_errors, delimiter=';', header='J;Ct;Cp;Eta;Ct_error;Cp_error;Eta_error', comments='')

# %%

# Plot CT, CP, and Eta with error bars over J
plt.figure(111)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

J_range = TestData['J']
mean_ct = TestData['Ct']
mean_cp = TestData['Cp']
mean_eta = TestData['eta']


# Plot CT
axs[0].errorbar(J_range, mean_ct, yerr=CT_errors, fmt='o', capsize=5, label='C_T')
axs[0].set_title('Thrust Coefficient (C_T) vs J', fontsize=14)
axs[0].set_xlabel('J', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].set_xlim(0.7, 1)
axs[0].set_ylim(0, 0.08)
axs[0].legend()

# Plot CP
axs[1].errorbar(J_range, mean_cp, yerr=CP_errors, fmt='o', capsize=5, label='C_P', color='red')
axs[1].set_title('Power Coefficient (C_P) vs J', fontsize=14)
axs[1].set_xlabel('J', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].set_xlim(0.7, 1)
axs[1].set_ylim(0, 0.08)
axs[1].legend()

# Plot Eta
axs[2].errorbar(J_range, mean_eta, yerr=Eta_errors, fmt='o', capsize=5, label='Eta', color='green')
axs[2].set_title('Efficiency (Eta) vs J', fontsize=14)
axs[2].set_xlabel('J', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].set_xlim(0.7, 1)
axs[2].set_ylim(0, 1)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


