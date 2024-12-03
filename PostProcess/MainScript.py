# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d
from Taylor_error import Taylor_error_propagation
from sympy import symbols

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

# Get all filenames in the input directory
all_files = os.listdir(input_base_path)

# Filter out only the relevant test files
test_files = [f for f in all_files if f.startswith('T') and f.endswith('.txt')]


   
    
# Initialize lists to store data for combined plots
all_J_baseline = []
all_Ct_baseline = []
all_Cp_baseline = []
all_eta_baseline = []

all_J_severe = []
all_Ct_severe = []
all_Cp_severe = []
all_eta_severe = []

all_rpm_baseline = []
all_rpm_severe = []
all_thrust_baseline = []
all_thrust_severe = []
all_torque_baseline = []
all_torque_severe = []
all_rpm_baseline = []
all_rpm_severe = []


# Process and save each test file
for file_name in test_files:
    file_path = os.path.join(input_base_path, file_name)
    
    # Read the test data
    TestData = pd.read_csv(file_path, delimiter='\t')

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

    # Separate baseline and severe data
    if 'Baseline' in file_name:
        all_J_baseline.append(TestData['J'])
        all_Ct_baseline.append(TestData['Ct'])
        all_Cp_baseline.append(TestData['Cp'])
        all_eta_baseline.append(TestData['eta'])
        all_rpm_baseline.append(TestData['RPM'])
        all_thrust_baseline.append(TestData['Thrust'])
        all_torque_baseline.append(TestData['Torque'])
    elif 'Severe' in file_name:
        all_J_severe.append(TestData['J'])
        all_Ct_severe.append(TestData['Ct'])
        all_Cp_severe.append(TestData['Cp'])
        all_eta_severe.append(TestData['eta'])
        all_rpm_severe.append(TestData['RPM'])
        all_thrust_severe.append(TestData['Thrust'])
        all_torque_severe.append(TestData['Torque'])


    # Prepare data to save, trimming the first and last rows
    A = TestData[['J', 'Ct', 'Cp', 'eta']].iloc[1:-1].to_numpy()

    # Generate output file path with `_pp.txt` suffix
    output_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_pp.txt")
    
    # Save data to file
    np.savetxt(output_file_path, A, delimiter=';', header='J;Ct;Cp;Eta', comments='')


# Plot all baseline data in one figure
plt.figure(100)
for J, Ct, Cp in zip(all_J_baseline, all_Ct_baseline, all_Cp_baseline):
    plt.scatter(J, Ct, label='Ct Baseline')
    plt.scatter(J, Cp, label='Cp Baseline', marker='+')
plt.xlabel('J')
plt.xlim([0.4, 1])
plt.legend()
plt.title('Baseline Ct and Cp')
plt.show()

plt.figure(101)
for J, eta in zip(all_J_baseline, all_eta_baseline):
    plt.scatter(J, eta, label='Eta Baseline', marker='+')
plt.xlabel('J')
plt.legend()
plt.title('Baseline Eta')
plt.show()

# Plot all severe data in one figure
plt.figure(102)
for J, Ct, Cp in zip(all_J_severe, all_Ct_severe, all_Cp_severe):
    plt.scatter(J, Ct, label='Ct Severe')
    plt.scatter(J, Cp, label='Cp Severe', marker='+')
plt.xlabel('J')
plt.xlim([0.4, 1])
plt.legend()
plt.title('Severe Ct and Cp')
plt.show()

plt.figure(103)
for J, eta in zip(all_J_severe, all_eta_severe):
    plt.scatter(J, eta, label='Eta Severe', marker='+')
plt.xlabel('J')
plt.legend()
plt.title('Severe Eta')
plt.show()

# Plot all data in one figure
plt.figure(104)
for J, Ct, Cp in zip(all_J_baseline + all_J_severe, all_Ct_baseline + all_Ct_severe, all_Cp_baseline + all_Cp_severe):
    plt.scatter(J, Ct, label='Ct All')
    plt.scatter(J, Cp, label='Cp All', marker='+')
plt.xlabel('J')
plt.xlim([0.4, 1])
plt.legend()
plt.title('All Ct and Cp')
plt.show()

plt.figure(105)
for J, eta in zip(all_J_baseline + all_J_severe, all_eta_baseline + all_eta_severe):
    plt.scatter(J, eta, label='Eta All', marker='+')
plt.xlabel('J')
plt.legend()
plt.title('All Eta')
plt.show()

# %%

rpm_range = np.linspace(6000, 8500, 20)

interpolated_thrust_baseline = []
interpolated_thrust_severe = []

for thrust, rpm in zip(all_thrust_baseline, all_rpm_baseline):
    f = interp1d(rpm[2:-1], thrust[2:-1], kind='linear', fill_value="extrapolate")
    interpolated_thrust_baseline.append(f(rpm_range))

for thrust, rpm in zip(all_thrust_severe, all_rpm_severe):
    f = interp1d(rpm[2:-1], thrust[2:-1], kind='linear', fill_value="extrapolate")
    interpolated_thrust_severe.append(f(rpm_range))   


# Plot interpolated thrust data and compare with sample data

plt.figure(106)
for thrust in interpolated_thrust_baseline:
    plt.plot(rpm_range, thrust, label='Thrust Baseline')
for thrust, rpm in zip(all_thrust_baseline, all_rpm_baseline):
    plt.scatter(rpm[2:-1], thrust[2:-1], marker='x')
plt.xlabel('RPM')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Baseline Thrust')
plt.show()


plt.figure(107)
for thrust in interpolated_thrust_severe:
    plt.plot(rpm_range, thrust, label='Thrust Severe')
for thrust, rpm in zip(all_thrust_severe, all_rpm_severe):
    plt.scatter(rpm[2:-1], thrust[2:-1], marker='x')
plt.xlabel('RPM')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Severe Thrust')
plt.show()

# %%
mean_thrust_baseline = np.mean(interpolated_thrust_baseline, axis=0)
mean_thrust_severe = np.mean(interpolated_thrust_severe, axis=0)

std_thrust_baseline = np.std(interpolated_thrust_baseline, axis=0)
std_thrust_severe = np.std(interpolated_thrust_severe, axis=0)

# Plot mean baseline thrust with uncertainty
plt.figure(108)
plt.plot(rpm_range, mean_thrust_baseline, label='Mean Thrust Baseline')
plt.fill_between(rpm_range, mean_thrust_baseline - std_thrust_baseline, mean_thrust_baseline + std_thrust_baseline, alpha=0.2)
plt.xlabel('RPM')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Baseline Mean Thrust with Uncertainty')
plt.show()

# Plot mean severe thrust with uncertainty
plt.figure(109)
plt.plot(rpm_range, mean_thrust_severe, label='Mean Thrust Severe')
plt.fill_between(rpm_range, mean_thrust_severe - std_thrust_severe, mean_thrust_severe + std_thrust_severe, alpha=0.2)
plt.xlabel('RPM')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Severe Mean Thrust with Uncertainty')
plt.show()



# %%

# Convert RPM range to advance ratio and plot mean thrust for baseline and severe cases with uncertainty
plt.figure(110)
J_range = 23 / ((rpm_range / 60) * Dia)  # Advance ratio
plt.plot(J_range, mean_thrust_baseline, label='Mean Thrust Baseline')
plt.fill_between(J_range, mean_thrust_baseline - std_thrust_baseline, mean_thrust_baseline + std_thrust_baseline, alpha=0.2)
plt.plot(J_range, mean_thrust_severe, label='Mean Thrust Severe')
plt.fill_between(J_range, mean_thrust_severe - std_thrust_severe, mean_thrust_severe + std_thrust_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Mean Thrust with Uncertainty')
plt.show()

# %%
interpolated_torque_baseline = []
interpolated_torque_severe = []

for torque, rpm in zip(all_torque_baseline, all_rpm_baseline):
    f = interp1d(rpm[2:-1], torque[2:-1], kind='linear', fill_value="extrapolate")
    interpolated_torque_baseline.append(f(rpm_range))

for torque, rpm in zip(all_torque_severe, all_rpm_severe):
    f = interp1d(rpm[2:-1], torque[2:-1], kind='linear', fill_value="extrapolate")
    interpolated_torque_severe.append(f(rpm_range))   


# Plot interpolated torque data and compare with sample data

plt.figure(106)
for torque in interpolated_torque_baseline:
    plt.plot(rpm_range, torque, label='Torque Baseline')
for torque, rpm in zip(all_torque_baseline, all_rpm_baseline):
    plt.scatter(rpm[2:-1], torque[2:-1], marker='x')
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Baseline Torque')
plt.show()


plt.figure(107)
for torque in interpolated_torque_severe:
    plt.plot(rpm_range, torque, label='Torque Severe')
for torque, rpm in zip(all_torque_severe, all_rpm_severe):
    plt.scatter(rpm[2:-1], torque[2:-1], marker='x')
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Severe Torque')
plt.show()

# %%
mean_torque_baseline = np.mean(interpolated_torque_baseline, axis=0)
mean_torque_severe = np.mean(interpolated_torque_severe, axis=0)

std_torque_baseline = np.std(interpolated_torque_baseline, axis=0)
std_torque_severe = np.std(interpolated_torque_severe, axis=0)

# Plot mean baseline torque with uncertainty
plt.figure(108)
plt.plot(rpm_range, mean_torque_baseline, label='Mean Torque Baseline')
plt.fill_between(rpm_range, mean_torque_baseline - std_torque_baseline, mean_torque_baseline + std_torque_baseline, alpha=0.2)
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Baseline Mean Torque with Uncertainty')
plt.show()

# Plot mean severe torque with uncertainty
plt.figure(109)
plt.plot(rpm_range, mean_torque_severe, label='Mean Torque Severe')
plt.fill_between(rpm_range, mean_torque_severe - std_torque_severe, mean_torque_severe + std_torque_severe, alpha=0.2)
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Severe Mean Torque with Uncertainty')
plt.show()



# %%

# Convert RPM range to advance ratio and plot mean torque for baseline and severe cases with uncertainty
plt.figure(110)
J_range = 23 / ((rpm_range / 60) * Dia)  # Advance ratio
plt.plot(J_range, mean_torque_baseline, label='Mean Torque Baseline')
plt.fill_between(J_range, mean_torque_baseline - std_torque_baseline, mean_torque_baseline + std_torque_baseline, alpha=0.2)
plt.plot(J_range, mean_torque_severe, label='Mean Torque Severe')
plt.fill_between(J_range, mean_torque_severe - std_torque_severe, mean_torque_severe + std_torque_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Mean Torque with Uncertainty')
plt.show()



# %%

mean_rpm_baseline = np.mean([np.interp(rpm_range, rpm[2:-1], rpm[2:-1]) for rpm in all_rpm_baseline], axis=0)
mean_rpm_severe = np.mean([np.interp(rpm_range, rpm[2:-1], rpm[2:-1]) for rpm in all_rpm_severe], axis=0)

std_rpm_baseline = np.std([np.interp(rpm_range, rpm[2:-1], rpm[2:-1]) for rpm in all_rpm_baseline], axis=0)
std_rpm_severe = np.std([np.interp(rpm_range, rpm[2:-1], rpm[2:-1]) for rpm in all_rpm_severe], axis=0)

# Plot mean baseline rpm with uncertainty
plt.figure(108)
plt.plot(rpm_range, mean_rpm_baseline, label='Mean RPM Baseline')
plt.fill_between(rpm_range, mean_rpm_baseline - std_rpm_baseline, mean_rpm_baseline + std_rpm_baseline, alpha=0.2)
plt.xlabel('RPM')
plt.ylabel('RPM')
plt.legend()
plt.title('Baseline Mean RPM with Uncertainty')
plt.show()




# %%

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

for i in range(len(mean_rpm_baseline)):
    # Sensor uncertainties
    uncertainties = {
        'T': std_rpm_baseline[i],  # Thrust uncertainty (N)
        'rho': 0,                 # Air density uncertainty
        'n': std_rpm_baseline[i],             # RPM uncertainty
        'd': 0.0003,              # Diameter uncertainty (m)
        'Q': std_torque_baseline[i],  # Torque uncertainty (Nm)
        'omega': 0,               # Angular velocity uncertainty (rad/s)
        'V': 0.1                  # Air velocity uncertainty (m/s)
    }
    T_value = mean_thrust_baseline[i]
    Q_value = mean_torque_baseline[i]
    n_value = mean_rpm_baseline[i] / 60
    rho_value = 1.225
    omega_value = 10

    # Define variable inputs for the Taylor propagation
    dict_variables_input = {
        'Variables': [T, rho, n, d, Q, omega, V],
        'Values': [T_value, rho_value, n_value, Dia, Q_value, omega_value, 23],
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
# %%

# Plot CT, CP, and Eta with error bars
plt.figure(111)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot CT
axs[0].errorbar(mean_rpm_baseline, mean_thrust_baseline, yerr=CT_errors, fmt='o', capsize=5, label='C_T')
axs[0].set_title('Thrust Coefficient (C_T) vs RPM', fontsize=14)
axs[0].set_xlabel('RPM', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].legend()

# Plot CP
axs[1].errorbar(mean_rpm_baseline, mean_torque_baseline, yerr=CP_errors, fmt='o', capsize=5, label='C_P', color='red')
axs[1].set_title('Power Coefficient (C_P) vs RPM', fontsize=14)
axs[1].set_xlabel('RPM', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].legend()

# Plot Eta
axs[2].errorbar(mean_rpm_baseline, mean_eta_baseline, yerr=Eta_errors, fmt='o', capsize=5, label='Eta', color='green')
axs[2].set_title('Efficiency (Eta) vs RPM', fontsize=14)
axs[2].set_xlabel('RPM', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()

# %%
