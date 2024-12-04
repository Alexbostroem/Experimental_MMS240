
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Read in CFD data of baseline to a pandas dataframe
df_baseline= pd.read_csv('cfd_data_baseline_sweep.csv')

df_baseline.head()

df_baseline.columns = [
    "RPM", "Cp", "RPM_eta", "Eta", 
    "RPM_thrust", "Thrust", 
    "RPM_power", "Power", 
    "RPM_ct", "Ct", "RPM_J", "J", 
    "RPM_torque", "Torque"
]
df_baseline = df_baseline.drop(columns=[col for col in df_baseline.columns if col.startswith('RPM_')])
df_baseline.head()
# %%


plt.plot(df_baseline['J'], df_baseline['Eta'], label='Baseline')


# %%
