# %% Kalibrering
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
from collections import defaultdict

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

# Process each test file and store data for cumulative plotting
for file in test_files:
    try:
        test_data = process_test_data(file, p, pT)


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
# Create DataFrames to store Cp for all tests
baseline_data_cp = pd.DataFrame()
severe_data_cp = pd.DataFrame()

for file in output_directory.glob("T*_pp.txt"):
    test_data = pd.read_csv(file, delimiter=';')
    if 'Baseline' in file.stem:
        baseline_data_cp[file.stem] = test_data['Cp']
    else:
        severe_data_cp[file.stem] = test_data['Cp']

# calculate mean and standard deviation for cp for each sample index in baseline and severe data
baseline_data_cp['mean'] = baseline_data_cp.mean(axis=1)
baseline_data_cp['std'] = baseline_data_cp.std(axis=1)
severe_data_cp['mean'] = severe_data_cp.mean(axis=1)
severe_data_cp['std'] = severe_data_cp.std(axis=1)

# Plot Cp for baseline and severe data with sample points and shaded areas for standard deviation
plt.figure(figsize=(10, 6))
plt.scatter(severe_data_cp.index, severe_data_cp['mean'], label='Severe', marker='o')
plt.fill_between(severe_data_cp.index, severe_data_cp['mean'] - severe_data_cp['std'], severe_data_cp['mean'] + severe_data_cp['std'], color='grey', alpha=0.3)
plt.scatter(baseline_data_cp.index, baseline_data_cp['mean'], label='Baseline', marker='x')
plt.fill_between(baseline_data_cp.index, baseline_data_cp['mean'] - baseline_data_cp['std'], baseline_data_cp['mean'] + baseline_data_cp['std'], color='grey', alpha=0.3)
plt.xlabel('Sample Index')
plt.ylabel('Cp')
plt.xlim(2, 13)
plt.title('Comparison of Cp Across Baseline and Severe Data')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# %%
# Create DataFrames to store Ct for all tests
baseline_data_ct = pd.DataFrame()
severe_data_ct = pd.DataFrame()

for file in output_directory.glob("T*_pp.txt"):
    test_data = pd.read_csv(file, delimiter=';')
    if 'Baseline' in file.stem:
        baseline_data_ct[file.stem] = test_data['Ct']
    else:
        severe_data_ct[file.stem] = test_data['Ct']

# calculate mean and standard deviation for ct for each sample index in baseline and severe data
baseline_data_ct['mean'] = baseline_data_ct.mean(axis=1)
baseline_data_ct['std'] = baseline_data_ct.std(axis=1)
severe_data_ct['mean'] = severe_data_ct.mean(axis=1)
severe_data_ct['std'] = severe_data_ct.std(axis=1)

# Plot Ct for baseline and severe data with sample points and shaded areas for standard deviation
plt.figure(figsize=(10, 6))
plt.scatter(severe_data_ct.index, severe_data_ct['mean'], label='Severe', marker='o')
plt.fill_between(severe_data_ct.index, severe_data_ct['mean'] - severe_data_ct['std'], severe_data_ct['mean'] + severe_data_ct['std'], color='grey', alpha=0.3)
plt.scatter(baseline_data_ct.index, baseline_data_ct['mean'], label='Baseline', marker='x')
plt.fill_between(baseline_data_ct.index, baseline_data_ct['mean'] - baseline_data_ct['std'], baseline_data_ct['mean'] + baseline_data_ct['std'], color='grey', alpha=0.3)
plt.xlabel('Sample Index')
plt.ylabel('Cp')
plt.xlim(2, 13)
plt.title('Comparison of Ct Across Baseline and Severe Data')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# %%
# %%
# Create DataFrames to store Eta for all tests
baseline_data_eta = pd.DataFrame()
severe_data_eta = pd.DataFrame()

for file in output_directory.glob("T*_pp.txt"):
    test_data = pd.read_csv(file, delimiter=';')
    if 'Baseline' in file.stem:
        baseline_data_eta[file.stem] = test_data['eta']
    else:
        severe_data_eta[file.stem] = test_data['eta']

# calculate mean and standard deviation for eta for each sample index in baseline and severe data
baseline_data_eta['mean'] = baseline_data_eta.mean(axis=1)
baseline_data_eta['std'] = baseline_data_eta.std(axis=1)
severe_data_eta['mean'] = severe_data_eta.mean(axis=1)
severe_data_eta['std'] = severe_data_eta.std(axis=1)

# Plot Eta for baseline and severe data with sample points and shaded areas for standard deviation
plt.figure(figsize=(10, 6))
plt.scatter(severe_data_eta.index, severe_data_eta['mean'], label='Severe', marker='o')
plt.fill_between(severe_data_eta.index, severe_data_eta['mean'] - severe_data_eta['std'], severe_data_eta['mean'] + severe_data_eta['std'], color='grey', alpha=0.3)
plt.scatter(baseline_data_eta.index, baseline_data_eta['mean'], label='Baseline', marker='x')
plt.fill_between(baseline_data_eta.index, baseline_data_eta['mean'] - baseline_data_eta['std'], baseline_data_eta['mean'] + baseline_data_eta['std'], color='grey', alpha=0.3)
plt.xlabel('Sample Index')
plt.ylabel('Eta')
plt.xlim(2, 13)
plt.ylim(0, 1)
plt.title('Comparison of Eta Across Baseline and Severe Data')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# %%
