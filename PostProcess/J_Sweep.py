# Define constants
target_J = 0.65
D = 0.236  # Propeller diameter in meters

# Calculate RPM for different wind speeds
wind_speeds = [10,12,15,17,20]  # in m/s
rpm_values = []

# J = U / (n*D)

for U in wind_speeds:
    n_required = 1/(target_J * (D / U))  # Required revolutions per second
    rpm_required = n_required * 60  # Convert to RPM
    print(f"Required RPM at {U} m/s: {rpm_required:.2f}")