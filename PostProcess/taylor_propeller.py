import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols
from Taylor_error import Taylor_error_propagation

# Close any open plots
plt.close('all')

# Set base paths for input and output files
input_base_path = r"Carbon"
output_directory = r"Carbon\PP"

# Read Torque calibration data
CalDataNm = pd.read_csv(f"{input_base_path}\\20241110_TorqueCalibration.txt", delimiter='\t')

# Reference values for calibration and adjustments
refCentre1 = CalDataNm['LoadL'][1]
refCentre2 = CalDataNm['LoadR'][1]
CalDataNm['LoadL'] -= refCentre1
CalDataNm['LoadR'] -= refCentre2

# Calibration parameters
RefPoints = np.array([0.2, 0.1, 0.05, 0, -0.05, -0.1]) * 9.82 * 0.1
NmCalc = 0.019 * (CalDataNm['LoadL'] + CalDataNm['LoadR'])
CalPoints = NmCalc[[4,3,2,1,6,8]]
p = np.polyfit(CalPoints, RefPoints, 1)

# Read Thrust calibration data
CalDataT = pd.read_csv(f"{input_base_path}\\241110-ThrustCalibration.txt", delimiter='\t')
RefPointsT = np.array([0,0.1,0.2,0.5,0,0.1,0.2,0.5]) * 9.82
TMeas = CalDataT['Thrust']
CalPointsT = TMeas[[0,1,2,3,4,5,6,7]]
pT = np.polyfit(CalPointsT, RefPointsT, 1)


# Process and save each test file
# Construct the specific input file path for each iteration
file_path = f"{input_base_path}\\241110_Carbon17ms_{5}.txt"
                    
# Read the test data
TestData = pd.read_csv(file_path, delimiter='\t')

# Calculate torque and other derived metrics
TestData['torque'] = abs(0.019 * (TestData['LoadL'] + TestData['LoadR']) * p[0])
Dia = 0.236
TestData['n'] = TestData['RPM'] / 60  # Revolutions per second
rho = TestData['rho']
TestData['J'] = TestData['U'] / (TestData['n'] * Dia)
TestData['Thrust'] = TestData['Thrust'] * pT[0]
TestData['Ct'] = TestData['Thrust'] / (rho * (TestData['n'] ** 2) * (Dia ** 4))
TestData['P'] = 2 * np.pi * TestData['n'] * TestData['torque']
TestData['Cp'] = TestData['P'] / (rho * (TestData['n'] ** 3) * (Dia ** 5))
TestData['eta'] = TestData['J'] * TestData['Ct'] / TestData['Cp']

# Define symbols
T, rho, n, d, Q, omega, V = symbols('T rho n d Q omega V')

# Define expressions for CT, CP, J, and eta
CT_Expr = T / (rho * n**2 * d**4)
CP_Expr = (Q * omega) / (rho * n**3 * d**5)
J_Expr = V / (n * d)
Eta_Expr = (CT_Expr * J_Expr) / CP_Expr


# Extract variables from the data
T_values = TestData["Thrust"]  # Thrust (N)
Q_values = TestData["Torque"]  # Torque (Nm)
RPS_values = TestData["n"]   # Rotational speed 
rho_values = TestData["rho"]   # Air density (kg/m^3)
CT_values = TestData['Ct']
CP_values = TestData['Cp']
Eta_values = TestData['eta']
J_num =  TestData['J']
d_value = 0.236           # Propeller diameter (m), assumed constant
V_value = 17               # Air velocity (m/s), assumed constant

# Convert RPM to n (revolutions per second) and omega (angular velocity)
n_values = RPS_values   # Revolutions per second
omega_values = 2 * np.pi * n_values  # Angular velocity (rad/s)

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
    'T': thrust_uncertainty,  # Thrust uncertainty (N)
    'rho': 0,                 # Air density uncertainty
    'n': 10 / 60,             # RPM uncertainty
    'd': 0.0003,              # Diameter uncertainty (m)
    'Q': torque_uncertainty,  # Torque uncertainty (Nm)
    'omega': 5,               # Angular velocity uncertainty (rad/s)
    'V': 0.1                  # Air velocity uncertainty (m/s)
}

# Loop through data points to compute CT, CP, Eta, and uncertainties

for i in range(len(TestData)):
    T_value = T_values[i]  
    Q_value = Q_values[i]
    n_value = n_values[i]
    rho_value = rho_values[i]
    omega_value =omega_values[i]

    # Define variable inputs for the Taylor propagation
    dict_variables_input = {
        'Variables': [T, rho, n, d, Q, omega, V],
        'Values': [T_value, rho_value, n_value, d_value, Q_value, omega_value, V_value],
        'Error_type': ['abs', 'abs', 'abs', 'abs', 'abs', 'abs', 'abs'],
        'Error': [uncertainties['T'], uncertainties['rho'], uncertainties['n'],
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

# Plot CT, CP, and Eta with error bars
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot CT
axs[0].errorbar(J_num, CT_values, yerr=CT_errors, fmt='o', capsize=5, label='C_T')
axs[0].set_title('Thrust Coefficient (C_T) vs RPM', fontsize=14)
axs[0].set_xlabel('RPM', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].legend()

# Plot CP
axs[1].errorbar(J_num, CP_values, yerr=CP_errors, fmt='o', capsize=5, label='C_P', color='red')
axs[1].set_title('Power Coefficient (C_P) vs RPM', fontsize=14)
axs[1].set_xlabel('RPM', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].legend()

# Plot Eta
axs[2].errorbar(J_num, Eta_values, yerr=Eta_errors, fmt='o', capsize=5, label='Efficiency (Eta)', color='green')
axs[2].set_title('Efficiency (Eta) vs RPM', fontsize=14)
axs[2].set_xlabel('RPM', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
