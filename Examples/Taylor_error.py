#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:18:06 2020
Taylor Expansion tool converted from Isak Jonssons .m file
@author: sudharsanvasudevan

2024/10/24: 
    Isak added the calculated_value output, nice to have

"""
# Import necessary packages
import numpy as np
import sympy as sp
import pandas as pd

# Define the function for Taylor error propagation
def Taylor_error_propagation(Phi, dict_variables_input):
    # Convert the input dictionary to a pandas DataFrame
    df_variables_input = pd.DataFrame(dict_variables_input, columns=['Variables', 'Values', 'Error_type', 'Error'])

    # Extract the variables, values, error types, and errors from the DataFrame
    variables = df_variables_input['Variables'].values.tolist()
    values = df_variables_input['Values'].values.tolist()
    Error_type = df_variables_input['Error_type'].values.tolist()
    Error = df_variables_input['Error'].values.tolist()

    # Initialize an empty list to store the derivatives
    derivative_individual = []

    # Compute the derivative of the expression with respect to each variable
    for var in variables:
        derivative_individual.append(sp.diff(Phi, var))

    # Initialize an empty list to store the individual uncertainties
    uncertainties_individual = []

    # Compute the uncertainty in the estimation of the desired quantity due to each variable
    for var1 in range(len(derivative_individual)):
        uncertainties_individual.append(derivative_individual[var1])
        for var2 in range(len(variables)):
            uncertainties_individual[var1] = uncertainties_individual[var1].subs(variables[var2], values[var2])
        if Error_type[var1] == 'abs':
            uncertainties_individual[var1] *= Error[var1]
        else:
            uncertainties_individual[var1] *= values[var1] * 0.01 * Error[var1]
        uncertainties_individual[var1] = np.abs(uncertainties_individual[var1])

    # Compute the total uncertainty as the square root of the sum of the squares of the individual uncertainties
    sum_uncertainty = (np.sum((np.array(uncertainties_individual))**2))**0.5

    # Create a list of variable names for the errors
    variable_names_error = []
    for var3 in variables:
        variable_names_error.append(sp.Symbol('( \delta '+str(var3)+' )'))

    # Create the final symbolic expression for the total uncertainty
    Expr_uncert = (variable_names_error[0]*derivative_individual[0])**2
    for var4 in range(1, len(variables)):
        Expr_uncert += (variable_names_error[var4]*derivative_individual[var4])**2
    Expr_uncert = sp.sqrt(Expr_uncert)

    # Substitute the values into the expression to get the calculated value of Phi
    calculated_value = Phi
    for var5 in range(len(variables)):
        calculated_value = calculated_value.subs(variables[var5], values[var5])
    
    # Evaluate the calculated value as a float
    calculated_value = float(calculated_value)

    # Return the individual uncertainties, the total uncertainty, and the symbolic expression for the total uncertainty
    return uncertainties_individual, sum_uncertainty, Expr_uncert, calculated_value