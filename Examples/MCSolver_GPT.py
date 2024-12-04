#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:18:06 2023
Monte Carlo tool directly from ChatGPT, very simple
@author: sudharsanvasudevan
"""
# Import necessary packages
import numpy as np
import sympy as sp
import pandas as pd

# Define the function for Monte Carlo uncertainty estimation
def MonteCarlo_error_propagation(Phi, dict_variables_input, num_samples):
    # Convert the input dictionary to a pandas DataFrame
    df_variables_input = pd.DataFrame(dict_variables_input, columns=['Variables', 'Values', 'Error_type', 'Error'])

    # Extract the variables, values, error types, and errors from the DataFrame
    variables = df_variables_input['Variables'].values
    values = df_variables_input['Values'].values
    Error_type = df_variables_input['Error_type'].values
    Error = df_variables_input['Error'].values

    # Generate random samples for each variable based on its uncertainty
    samples = []
    for i in range(len(variables)):
        if Error_type[i] == 'abs':
            samples.append(np.random.normal(values[i], Error[i], (num_samples,)))
        else:
            samples.append(np.random.normal(values[i], values[i] * 0.01 * Error[i], (num_samples,)))

    # Stack the samples to create a matrix
    samples_matrix = np.column_stack(samples)

    # Evaluate the expression for each set of random samples
    results = np.array([float(Phi.subs(dict(zip(variables, sample)))) for sample in samples_matrix])

    # Compute the mean and standard deviation of the results
    mean_result = np.mean(results)
    std_result = np.std(results)

    # Return the mean result and standard deviation as the uncertainty estimate
    return mean_result, std_result
