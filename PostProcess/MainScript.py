# %%
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
output_directory = r"2024_11_27\PP"

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

input_base_path = Path(input_base_path)


# List all test files, excluding specific ones
exclude_files = {'thrust_calib_baseline.txt', 'torque_calib_baseline.txt'}
test_files = sorted(
    file for file in input_base_path.glob("T*_*.txt") if file.name not in exclude_files
)


# %%

# Function to process test data
def process_test_data(file_path, p_torque, p_thrust):
    test_data = pd.read_csv(file_path, delimiter='\t')


    # Calculate derived metrics
    test_data['torque'] = abs(0.019 * (test_data['LoadL'] + test_data['LoadR']) * p_torque[0])
    dia = 0.276
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
        result.to_csv(output_directory / f"{file.stem}_pp.csv", sep=';', index=False)
    
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

baseline_ct = defaultdict(list)
baseline_cp = defaultdict(list)
baseline_eta = defaultdict(list)

severe_ct = defaultdict(list)
severe_cp = defaultdict(list)
severe_eta = defaultdict(list)



for file in test_files:
    try:
        # Process the test data
        test_data = process_test_data(file, p, pT)

        # Extract parts of the filename
        parts = file.stem.split('_')
        scenario_name = f"{'_'.join(parts[:2])}_{parts[-1]}"

        # Determine whether the data belongs to baseline or severe
        if 'Baseline' in parts:
             for idx, row in test_data.iterrows():
                baseline_ct[idx].append(row['Ct'])
                baseline_cp[idx].append(row['Cp'])
                baseline_eta[idx].append(row['eta'])
        elif 'Severe' in parts:
                for idx, row in test_data.iterrows():
                    severe_ct[idx].append(row['Ct'])
                    severe_cp[idx].append(row['Cp'])
                    severe_eta[idx].append(row['eta'])

    except Exception as e:
        print(f"Error processing file {file}: {e}")



# %% 
# Function to compute mean and std for a dictionary of lists
def compute_stats(data_dict):
    stats = {}
    for idx, values in data_dict.items():
        stats[idx] = {'mean': np.mean(values), 'std': np.std(values)}
    return stats

# Compute statistics for baseline and severe datasets
baseline_ct_stats = compute_stats(baseline_ct)
baseline_cp_stats = compute_stats(baseline_cp)
baseline_eta_stats = compute_stats(baseline_eta)

severe_ct_stats = compute_stats(severe_ct)
severe_cp_stats = compute_stats(severe_cp)
severe_eta_stats = compute_stats(severe_eta)

print(baseline_cp_stats)

# %%
# Convert stats dictionaries to DataFrames for easier analysis and visualization
baseline_stats_df = pd.DataFrame({
    'Index': baseline_ct_stats.keys(),
    'Ct Mean': [baseline_ct_stats[idx]['mean'] for idx in baseline_ct_stats],
    'Ct Std': [baseline_ct_stats[idx]['std'] for idx in baseline_ct_stats],
    'Cp Mean': [baseline_cp_stats[idx]['mean'] for idx in baseline_cp_stats],
    'Cp Std': [baseline_cp_stats[idx]['std'] for idx in baseline_cp_stats],
    'Eta Mean': [baseline_eta_stats[idx]['mean'] for idx in baseline_eta_stats],
    'Eta Std': [baseline_eta_stats[idx]['std'] for idx in baseline_eta_stats],
}).set_index('Index')

severe_stats_df = pd.DataFrame({
    'Index': severe_ct_stats.keys(),
    'Ct Mean': [severe_ct_stats[idx]['mean'] for idx in severe_ct_stats],
    'Ct Std': [severe_ct_stats[idx]['std'] for idx in severe_ct_stats],
    'Cp Mean': [severe_cp_stats[idx]['mean'] for idx in severe_cp_stats],
    'Cp Std': [severe_cp_stats[idx]['std'] for idx in severe_cp_stats],
    'Eta Mean': [severe_eta_stats[idx]['mean'] for idx in severe_eta_stats],
    'Eta Std': [severe_eta_stats[idx]['std'] for idx in severe_eta_stats],
}).set_index('Index')


# %%
# Plot Ct comparison
plt.figure(figsize=(10, 6))
plt.errorbar(
    baseline_stats_df.index,
    baseline_stats_df['Ct Mean'],
    yerr=baseline_stats_df['Ct Std'],
    fmt='o',
    label='Baseline Ct',
    capsize=3
)
plt.errorbar(
    severe_stats_df.index,
    severe_stats_df['Ct Mean'],
    yerr=severe_stats_df['Ct Std'],
    fmt='x',
    label='Severe Ct',
    capsize=3
)
plt.xlabel('Sample Index')
plt.ylabel('Ct')
plt.title('Comparison of Ct Between Baseline and Severe Datasets')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot Cp comparison
plt.figure(figsize=(10, 6))
plt.errorbar(
    baseline_stats_df.index,
    baseline_stats_df['Cp Mean'],
    yerr=baseline_stats_df['Cp Std'],
    fmt='o',
    label='Baseline Cp',
    capsize=3
)
plt.errorbar(
    severe_stats_df.index,
    severe_stats_df['Cp Mean'],
    yerr=severe_stats_df['Cp Std'],
    fmt='x',
    label='Severe Cp',
    capsize=3
)
plt.xlabel('Sample Index')
plt.ylabel('Cp')
plt.title('Comparison of Cp Between Baseline and Severe Datasets')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot Eta comparison
plt.figure(figsize=(10, 6))
plt.errorbar(
    baseline_stats_df.index,
    baseline_stats_df['Eta Mean'],
    yerr=baseline_stats_df['Eta Std'],
    fmt='o',
    label='Baseline Eta',
    capsize=3
)
plt.errorbar(
    severe_stats_df.index,
    severe_stats_df['Eta Mean'],
    yerr=severe_stats_df['Eta Std'],
    fmt='x',
    label='Severe Eta',
    capsize=3
)
plt.xlabel('Sample Index')
plt.ylabel('Eta')
plt.title('Comparison of Efficiency (Eta) Between Baseline and Severe Datasets')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

