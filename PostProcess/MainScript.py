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
all_temperature_baseline = []
all_pressure_baseline = []
all_temperature_severe = []
all_pressure_severe = []


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
        all_temperature_baseline.append(TestData['T'])
        all_pressure_baseline.append(TestData['Pa'])

    elif 'Severe' in file_name:
        all_J_severe.append(TestData['J'])
        all_Ct_severe.append(TestData['Ct'])
        all_Cp_severe.append(TestData['Cp'])
        all_eta_severe.append(TestData['eta'])
        all_rpm_severe.append(TestData['RPM'])
        all_thrust_severe.append(TestData['Thrust'])
        all_torque_severe.append(TestData['Torque'])
        all_temperature_severe.append(TestData['T'])
        all_pressure_severe.append(TestData['Pa'])


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


# Define the range to use for interpolation
range_start = 3
range_end = -1

rpm_range = np.linspace(6400, 8300, 22)


interpolated_thrust_baseline = []
interpolated_thrust_severe = []

for thrust, rpm in zip(all_thrust_baseline, all_rpm_baseline):
    f = interp1d(rpm[range_start:range_end], thrust[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_thrust_baseline.append(f(rpm_range))

for thrust, rpm in zip(all_thrust_severe, all_rpm_severe):
    f = interp1d(rpm[range_start:range_end], thrust[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_thrust_severe.append(f(rpm_range))   

# Plot interpolated thrust data and compare with sample data

plt.figure(106)
for thrust in interpolated_thrust_baseline:
    plt.plot(rpm_range, thrust, label='Thrust Baseline')
for thrust, rpm in zip(all_thrust_baseline, all_rpm_baseline):
    plt.scatter(rpm[range_start:range_end], thrust[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Baseline Thrust')
plt.show()

plt.figure(107)
for thrust in interpolated_thrust_severe:
    plt.plot(rpm_range, thrust, label='Thrust Severe')
for thrust, rpm in zip(all_thrust_severe, all_rpm_severe):
    plt.scatter(rpm[range_start:range_end], thrust[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Severe Thrust')
plt.show()

mean_thrust_baseline = np.mean(interpolated_thrust_baseline, axis=0)
mean_thrust_severe = np.mean(interpolated_thrust_severe, axis=0)

std_thrust_baseline = np.std(interpolated_thrust_baseline, axis=0)
std_thrust_severe = np.std(interpolated_thrust_severe, axis=0)


# Convert RPM range to advance ratio and plot mean thrust for baseline and severe cases with uncertainty
plt.figure(110)
J_range = 23 / ((rpm_range / 60) * Dia)  # Advance ratio
plt.plot(J_range, mean_thrust_baseline, label='Mean Thrust Baseline')
plt.plot(df_baseline['J'], df_baseline['Thrust'] *2 , label='CFD Thrust Baseline')
plt.plot(df_severe['J'], df_severe['Thrust'] *2, label='CFD Thrust Severe')
plt.fill_between(J_range, mean_thrust_baseline - std_thrust_baseline, mean_thrust_baseline + std_thrust_baseline, alpha=0.2)
plt.plot(J_range, mean_thrust_severe, label='Mean Thrust Severe')
plt.fill_between(J_range, mean_thrust_severe - std_thrust_severe, mean_thrust_severe + std_thrust_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Mean Thrust with Uncertainty')
plt.savefig(os.path.join(output_directory, 'Mean Thrust with Uncertainty.png'), dpi=300)
plt.show()

# %%
interpolated_torque_baseline = []
interpolated_torque_severe = []

for torque, rpm in zip(all_torque_baseline, all_rpm_baseline):
    f = interp1d(rpm[range_start:range_end], torque[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_torque_baseline.append(f(rpm_range))

for torque, rpm in zip(all_torque_severe, all_rpm_severe):
    f = interp1d(rpm[range_start:range_end], torque[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_torque_severe.append(f(rpm_range))   

# Plot interpolated torque data and compare with sample data

plt.figure(106)
for torque in interpolated_torque_baseline:
    plt.plot(rpm_range, torque, label='Torque Baseline')
for torque, rpm in zip(all_torque_baseline, all_rpm_baseline):
    plt.scatter(rpm[range_start:range_end], torque[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Baseline Torque')
plt.show()

plt.figure(107)
for torque in interpolated_torque_severe:
    plt.plot(rpm_range, torque, label='Torque Severe')
for torque, rpm in zip(all_torque_severe, all_rpm_severe):
    plt.scatter(rpm[range_start:range_end], torque[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Severe Torque')
plt.show()

mean_torque_baseline = np.mean(interpolated_torque_baseline, axis=0)
mean_torque_severe = np.mean(interpolated_torque_severe, axis=0)

std_torque_baseline = np.std(interpolated_torque_baseline, axis=0)
std_torque_severe = np.std(interpolated_torque_severe, axis=0)


# Convert RPM range to advance ratio and plot mean torque for baseline and severe cases with uncertainty
J_range = 23 / ((rpm_range / 60) * Dia)  # Advance ratio
plt.figure(110)
plt.plot(J_range, mean_torque_baseline, label='Mean Torque Baseline')
plt.plot(df_baseline['J'], df_baseline['Torque'], label='CFD Torque Baseline')
plt.plot(df_severe['J'], df_severe['Torque'], label='CFD Torque Severe')
plt.fill_between(J_range, mean_torque_baseline - std_torque_baseline, mean_torque_baseline + std_torque_baseline, alpha=0.2)
plt.plot(J_range, mean_torque_severe, label='Mean Torque Severe')
plt.fill_between(J_range, mean_torque_severe - std_torque_severe, mean_torque_severe + std_torque_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Mean Torque with Uncertainty')
plt.savefig(os.path.join(output_directory, 'Mean Torque with Uncertainty.png'), dpi=300)
plt.show()

# %%

mean_rpm_baseline = np.mean([np.interp(rpm_range, rpm[range_start:range_end], rpm[range_start:range_end]) for rpm in all_rpm_baseline], axis=0)
mean_rpm_severe = np.mean([np.interp(rpm_range, rpm[range_start:range_end], rpm[range_start:range_end]) for rpm in all_rpm_severe], axis=0)

std_rpm_baseline = np.std([np.interp(rpm_range, rpm[range_start:range_end], rpm[range_start:range_end]) for rpm in all_rpm_baseline], axis=0)
std_rpm_severe = np.std([np.interp(rpm_range, rpm[range_start:range_end], rpm[range_start:range_end]) for rpm in all_rpm_severe], axis=0)



#%% 
interpolated_eta_baseline = []
interpolated_eta_severe = []

for eta, rpm in zip(all_eta_baseline, all_rpm_baseline):
    f = interp1d(rpm[range_start:range_end], eta[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_eta_baseline.append(f(rpm_range))

for eta, rpm in zip(all_eta_severe, all_rpm_severe):
    f = interp1d(rpm[range_start:range_end], eta[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_eta_severe.append(f(rpm_range))   

plt.figure(106)
for eta in interpolated_eta_baseline:
    plt.plot(rpm_range, eta, label='Eta Baseline')
for eta, rpm in zip(all_eta_baseline, all_rpm_baseline):
    plt.scatter(rpm[range_start:range_end], eta[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Eta')
plt.legend()
plt.title('Baseline Eta')
plt.show()

plt.figure(107)
for eta in interpolated_eta_severe:
    plt.plot(rpm_range, eta, label='Eta Severe')
for eta, rpm in zip(all_eta_severe, all_rpm_severe):
    plt.scatter(rpm[range_start:range_end], eta[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Eta')
plt.legend()
plt.title('Severe Eta')
plt.show()

mean_eta_baseline = np.mean(interpolated_eta_baseline, axis=0)
mean_eta_severe = np.mean(interpolated_eta_severe, axis=0)

std_eta_baseline = np.std(interpolated_eta_baseline, axis=0)
std_eta_severe = np.std(interpolated_eta_severe, axis=0)

# Convert RPM range to advance ratio and plot mean eta for baseline and severe cases with uncertainty
plt.figure(110)
plt.plot(J_range, mean_eta_baseline, label='Mean Eta Baseline')
plt.plot(df_baseline['J'], df_baseline['Eta'], label='CFD Eta Baseline')
plt.plot(df_severe['J'], df_severe['Eta'], label='CFD Eta Severe')
plt.fill_between(J_range, mean_eta_baseline - std_eta_baseline, mean_eta_baseline + std_eta_baseline, alpha=0.2)
plt.plot(J_range, mean_eta_severe, label='Mean Eta Severe')
plt.fill_between(J_range, mean_eta_severe - std_eta_severe, mean_eta_severe + std_eta_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Eta')
plt.legend()
plt.title('Mean Eta with Uncertainty')
plt.savefig(os.path.join(output_directory, 'Mean Eta with Uncertainty.png'), dpi=300)
plt.show()

# %%

interpolated_Ct_baseline = []
interpolated_Ct_severe = []

for Ct, rpm in zip(all_Ct_baseline, all_rpm_baseline):
    f = interp1d(rpm[range_start:range_end], Ct[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_Ct_baseline.append(f(rpm_range))

for Ct, rpm in zip(all_Ct_severe, all_rpm_severe):
    f = interp1d(rpm[range_start:range_end], Ct[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_Ct_severe.append(f(rpm_range))   

plt.figure(106)
for Ct in interpolated_Ct_baseline:
    plt.plot(rpm_range, Ct, label='Ct Baseline')
for Ct, rpm in zip(all_Ct_baseline, all_rpm_baseline):
    plt.scatter(rpm[range_start:range_end], Ct[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Ct')
plt.legend()
plt.title('Baseline Ct')
plt.show()

plt.figure(107)
for Ct in interpolated_Ct_severe:
    plt.plot(rpm_range, Ct, label='Ct Severe')
for Ct, rpm in zip(all_Ct_severe, all_rpm_severe):
    plt.scatter(rpm[range_start:range_end], Ct[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Ct')
plt.legend()
plt.title('Severe Ct')
plt.show()

mean_Ct_baseline = np.mean(interpolated_Ct_baseline, axis=0)
mean_Ct_severe = np.mean(interpolated_Ct_severe, axis=0)

std_Ct_baseline = np.std(interpolated_Ct_baseline, axis=0)
std_Ct_severe = np.std(interpolated_Ct_severe, axis=0)


# Convert RPM range to advance ratio and plot mean eta for baseline and severe cases with uncertainty
plt.figure(110)
plt.plot(J_range, mean_Ct_baseline, label='Mean Ct Baseline')
plt.plot(df_baseline['J'], df_baseline['Ct'], label='CFD Ct Baseline')
plt.plot(df_severe['J'], df_severe['Ct'], label='CFD Ct Severe')
plt.fill_between(J_range, mean_Ct_baseline - std_Ct_baseline, mean_Ct_baseline + std_Ct_baseline, alpha=0.2)
plt.plot(J_range, mean_Ct_severe, label='Mean Ct Severe')
plt.fill_between(J_range, mean_Ct_severe - std_Ct_severe, mean_Ct_severe + std_Ct_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Ct')
plt.legend()
plt.title('Mean Ct with Uncertainty')
plt.savefig(os.path.join(output_directory, 'Mean Ct with Uncertainty.png'), dpi=300)
plt.show()


# %%

interpolated_Cp_baseline = []
interpolated_Cp_severe = []

for Cp, rpm in zip(all_Cp_baseline, all_rpm_baseline):
    f = interp1d(rpm[range_start:range_end], Cp[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_Cp_baseline.append(f(rpm_range))

for Cp, rpm in zip(all_Cp_severe, all_rpm_severe):
    f = interp1d(rpm[range_start:range_end], Cp[range_start:range_end], kind='linear', fill_value="extrapolate")
    interpolated_Cp_severe.append(f(rpm_range))   

plt.figure(106)
for Cp in interpolated_Cp_baseline:
    plt.plot(rpm_range, Cp, label='Cp Baseline')
for Cp, rpm in zip(all_Cp_baseline, all_rpm_baseline):
    plt.scatter(rpm[range_start:range_end], Cp[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Cp')
plt.legend()
plt.title('Baseline Cp')
plt.show()

plt.figure(107)
for Cp in interpolated_Cp_severe:
    plt.plot(rpm_range, Cp, label='Cp Severe')
for Cp, rpm in zip(all_Cp_severe, all_rpm_severe):
    plt.scatter(rpm[range_start:range_end], Cp[range_start:range_end], marker='x')
plt.xlabel('RPM')
plt.ylabel('Cp')
plt.legend()
plt.title('Severe Cp')
plt.show()

mean_Cp_baseline = np.mean(interpolated_Cp_baseline, axis=0)
mean_Cp_severe = np.mean(interpolated_Cp_severe, axis=0)

std_Cp_baseline = np.std(interpolated_Cp_baseline, axis=0)
std_Cp_severe = np.std(interpolated_Cp_severe, axis=0)

# Convert RPM range to advance ratio and plot mean eta for baseline and severe cases with uncertainty
plt.figure(110)
plt.plot(J_range, mean_Cp_baseline, label='Mean Cp Baseline')
plt.plot(df_baseline['J'], df_baseline['Cp'], label='CFD Cp Baseline')
plt.plot(df_severe['J'], df_severe['Cp'], label='CFD Cp Severe')
plt.fill_between(J_range, mean_Cp_baseline - std_Cp_baseline, mean_Cp_baseline + std_Cp_baseline, alpha=0.2)
plt.plot(J_range, mean_Cp_severe, label='Mean Cp Severe')
plt.fill_between(J_range, mean_Cp_severe - std_Cp_severe, mean_Cp_severe + std_Cp_severe, alpha=0.2)
plt.xlabel('J')
plt.ylabel('Cp')
plt.legend()
plt.title('Mean Cp with Uncertainty')
plt.savefig(os.path.join(output_directory, 'Mean Cp with Uncertainty.png'), dpi=300)
plt.show()


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

for i in range(len(mean_rpm_baseline)):
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

    T_value = mean_thrust_baseline[i]
    Q_value = mean_torque_baseline[i]
    n_value = mean_rpm_baseline[i] / 60
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

# Plot CT, CP, and Eta with error bars over J
plt.figure(111)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot CT
axs[0].errorbar(J_range, mean_Ct_baseline, yerr=CT_errors, fmt='o', capsize=5, label='C_T')
axs[0].fill_between(J_range, mean_Ct_baseline - std_Ct_baseline, mean_Ct_baseline + std_Ct_baseline, alpha=0.2)
axs[0].set_title('Thrust Coefficient (C_T) vs J', fontsize=14)
axs[0].set_xlabel('J', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].legend()

# Plot CP
axs[1].errorbar(J_range, mean_Cp_baseline, yerr=CP_errors, fmt='o', capsize=5, label='C_P', color='red')
axs[1].set_title('Power Coefficient (C_P) vs J', fontsize=14)
axs[1].fill_between(J_range, mean_Cp_baseline - std_Cp_baseline, mean_Cp_baseline + std_Cp_baseline, alpha=0.2)
axs[1].set_xlabel('J', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].legend()

# Plot Eta
axs[2].errorbar(J_range, mean_eta_baseline, yerr=Eta_errors, fmt='o', capsize=5, label='Eta', color='green')
axs[2].set_title('Efficiency (Eta) vs J', fontsize=14)
axs[2].fill_between(J_range, mean_eta_baseline - std_eta_baseline, mean_eta_baseline + std_eta_baseline, alpha=0.2)
axs[2].set_xlabel('J', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].set_ylim(0, 1)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


# %%
# Placeholder for computed values and errors
CT_errors_severe = []
CP_errors_severe = []
Eta_errors_severe = []

# Compute errors for severe data

# Thrust and torque sensor uncertainties
full_scale_thrust = 9.81  # Full scale in N (1 kg capacity = 9.81 N)
thrust_uncertainty = np.sqrt(
    (0.05 / 100 * full_scale_thrust)**2  # Non-linearity
    )

full_scale_torque = 1 * 9.81 * 0.019  # 1 kg capacity at 19 mm arm 

torque_uncertainty = np.sqrt(
    (0.05 / 100 * full_scale_torque)**2   # Non-linearity
    )


for i in range(len(mean_rpm_severe)):
    # Sensor uncertainties
    uncertainties = {
        'T': thrust_uncertainty,  # Thrust uncertainty (N) std_thrust_severe[i]
        'rho': 0.0037,              # Air density uncertainty
        'n': 10/60 ,              # RPS uncertainty std_rpm_severe[i]/60
        'd': 0.0003,              # Diameter uncertainty (m)
        'Q': torque_uncertainty,  # Torque uncertainty (Nm)
        'omega': 0,               # Angular velocity uncertainty (rad/s)
        'V': 0.1                  # Air velocity uncertainty (m/s)
    }

    T_value = mean_thrust_severe[i]
    Q_value = mean_torque_severe[i]
    n_value = mean_rpm_severe[i] / 60
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
    CT_errors_severe.append(CT_sum_uncert)
    CP_errors_severe.append(CP_sum_uncert)
    Eta_errors_severe.append(Eta_sum_uncert)

# %%

# Plot CT, CP, and Eta with error bars over J for severe data
plt.figure(111)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot CT
axs[0].errorbar(J_range, mean_Ct_severe, yerr=CT_errors_severe, fmt='o', capsize=5, label='C_T')
axs[0].fill_between(J_range, mean_Ct_severe - std_Ct_severe, mean_Ct_severe + std_Ct_severe, alpha=0.2)
axs[0].set_title('Thrust Coefficient (C_T) vs J', fontsize=14)
axs[0].set_xlabel('J', fontsize=12)
axs[0].set_ylabel('C_T', fontsize=12)
axs[0].grid(True)
axs[0].legend()

# Plot CP
axs[1].errorbar(J_range, mean_Cp_severe, yerr=CP_errors_severe, fmt='o', capsize=5, label='C_P', color='red')
axs[1].set_title('Power Coefficient (C_P) vs J', fontsize=14)
axs[1].fill_between(J_range, mean_Cp_severe - std_Cp_severe, mean_Cp_severe + std_Cp_severe, alpha=0.2)
axs[1].set_xlabel('J', fontsize=12)
axs[1].set_ylabel('C_P', fontsize=12)
axs[1].grid(True)
axs[1].legend()

# Plot Eta
axs[2].errorbar(J_range, mean_eta_severe, yerr=Eta_errors_severe, fmt='o', capsize=5, label='Eta', color='green')
axs[2].set_title('Efficiency (Eta) vs J', fontsize=14)
axs[2].fill_between(J_range, mean_eta_severe - std_eta_severe, mean_eta_severe + std_eta_severe, alpha=0.2)
axs[2].set_xlabel('J', fontsize=12)
axs[2].set_ylabel('Eta', fontsize=12)
axs[2].set_ylim(0, 1)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


# %%

# Plot Eta with error bars over J for baseline and severe data in same figure
plt.figure(113)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))

# Plot Eta
axs.errorbar(J_range, mean_eta_baseline, yerr=Eta_errors, fmt='o', capsize=5, label='Eta Baseline', color='green')

axs.errorbar(J_range, mean_eta_severe, yerr=Eta_errors_severe, fmt='o', capsize=5, label='Eta Severe', color='red')

axs.set_title('Efficiency (Eta) vs J', fontsize=14)
axs.set_xlabel('J', fontsize=12)
axs.set_ylabel('Eta', fontsize=12)
axs.set_ylim(0, 1)
axs.grid(True)
axs.legend()

# %%
