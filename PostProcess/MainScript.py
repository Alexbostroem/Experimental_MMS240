import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Close any open plots
plt.close('all')

# Set base paths for input and output files
input_base_path = r"Baseline"
output_directory = r"Baseline\PP"

# Read Torque calibration data
CalDataNm = pd.read_csv(f"{input_base_path}\\torque_calib.txt", delimiter='\t')

# Reference values for calibration and adjustments
refCentre1 = CalDataNm['LoadL'][1]
refCentre2 = CalDataNm['LoadR'][1]
CalDataNm['LoadL'] -= refCentre1
CalDataNm['LoadR'] -= refCentre2

# Calibration parameters
RefPoints = np.array([0, 0.05, 0.1, 0.2, -0.05, -0.1, -0.2] ) * 9.82 * 0.1
NmCalc = 0.019 * (CalDataNm['LoadL'] + CalDataNm['LoadR'])
CalPoints = NmCalc[[1 , 3 , 5 , 7 , 12 , 14, 16 ]]
p = np.polyfit(CalPoints, RefPoints, 1)

# Read Thrust calibration data
CalDataT = pd.read_csv(f"{input_base_path}\\thrust_calib.txt", delimiter='\t')
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

# Process and save each test file
for ii in range(1, 3):
    # Construct the specific input file path for each iteration
    file_path = f"{input_base_path}\\241110_Carbon17ms_{ii}.txt"
    
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

    # Plot data
    plt.figure(98)
    plt.scatter(TestData['J'], TestData['Ct'], label='Ct')
    plt.scatter(TestData['J'], TestData['Cp'], label='Cp', marker='+')
    plt.xlabel('J')
    plt.xlim([0.4, 1])
    plt.legend()
    plt.show()
    
    plt.figure(99)
    plt.scatter(TestData['J'], TestData['eta'], label='Eta', marker='+')
    plt.xlim([0.4, 1])
    plt.ylim([0.5, 0.8])
    plt.xlabel('J')
    plt.legend()
    plt.show()

    # Prepare data to save, trimming the first and last rows
    A = TestData[['J', 'Ct', 'Cp', 'eta']].iloc[1:-1].to_numpy()

    # Generate output file path with `_pp.txt` suffix
    output_file_path = f"{output_directory}\\baseline{ii}_pp.txt"
    
    # Save data to file
    np.savetxt(output_file_path, A, delimiter=';', header='J;Ct;Cp;Eta', comments='')
