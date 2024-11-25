#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary packages
import numpy as np
from sympy import symbols, Symbol
from Taylor_error import Taylor_error_propagation
from MCSolver_GPT import MonteCarlo_error_propagation

# Define the symbols for the variables
T = Symbol('T')
rho = Symbol('rho')
n = Symbol('n')
d = Symbol('d')

# Define the expression for which the Taylor expansion is to be performed
CT_Expr =  T / (rho * n**2 * d**4)

# Define a dictionary with the variables, their values, the type of error, and the error values
# 'Variables': List of variables used in the expression
# 'Values': Corresponding values of the variables
# 'Error_type': Type of error associated with each variable. It can be 'abs' or 'rel'.
#     'abs' means the error is absolute, i.e., the error value is a fixed quantity.
#     'rel' means the error is relative, i.e., the error value is a percentage of the variable's value.
# 'Error': The error values associated with each variable. If the error type is 'abs', this is an absolute quantity.
#          If the error type is 'rel', this is a percentage and should be multiplied by the variable's value to get the absolute error.
dict_variables_input_q ={'Variables': [T, rho, n , d],
                 'Values': [3.0829, 1.225, 4050/60, 0.3429],
                 'Error_type': ['abs','abs','abs','abs'],
                 'Error': [0.1, 0, 10/60 , 0.003]
                 }

# dict_variables_input_q ={'Variables': [T, rho, n , d],
#                   'Values': [3.924, 1.225, 4300/60, 0.3429],
#                   'Error_type': ['abs','abs','abs','abs'],
#                   'Error': [0.1, 0, 10/60 , 0.003]
#                   }

# Call the Taylor_error_propagation function with the expression and the dictionary as inputs
# The function performs a Taylor expansion of the first order on the given expression.
# It computes the derivative of the expression with respect to each variable, 
# calculates the uncertainties in the estimation of the desired quantity due to error in each of the individual variables,
# and finally computes the total uncertainty as the square root of the sum of the squares of the individual uncertainties.
# The function returns three outputs:
# 1. uncertainties: A list of the individual uncertainties for each variable.
# 2. sum_uncert: The total uncertainty, calculated as the square root of the sum of the squares of the individual uncertainties.
# 3. Expr_uncert: A symbolic expression for the total uncertainty.
uncertainties, sum_uncert, Expr_uncert, totUncert = Taylor_error_propagation(CT_Expr, dict_variables_input_q)
Value, uncertMC = MonteCarlo_error_propagation(CT_Expr, dict_variables_input_q,1000)

# Print the total uncertainty
print(sum_uncert)

# Print the total uncertainty
print(uncertMC)

# Print the ratio of individual uncertainties to the total uncertainty
print(np.divide(uncertainties, sum_uncert))

# Print the ratio of the total uncertainty to the calculated expression
print(sum_uncert/CT_Expr)


F = Symbol('F')
r = Symbol('r')

# Define the expression for which the Taylor expansion is to be performed
T_Expr = 2*F*r 

#  Rotational center, defined load to be positive around rotation
#                   | 100g or 0.982 grams
#+ Load1            +
#|%%%%%%%%%%%%%%%%%(X)%%%%%%%%%%%%%%%%|
#                                   + Load2

# Define a dictionary with the variables, their values, the type of error, and the error values
# 'Variables': List of variables used in the expression
# 'Values': Corresponding values of the variables
# 'Error_type': Type of error associated with each variable. It can be 'abs' or 'rel'.
#     'abs' means the error is absolute, i.e., the error value is a fixed quantity.
#     'rel' means the error is relative, i.e., the error value is a percentage of the variable's value.
# 'Error': The error values associated with each variable. If the error type is 'abs', this is an absolute quantity.
#          If the error type is 'rel', this is a percentage and should be multiplied by the variable's value to get the absolute error.
dict_variables ={'Variables': [F, r],
                 'Values': [2.492105263157895,0.019],
                 'Error_type': ['abs','abs'],
                 'Error': [10*0.01, 0]
                 }

# dict_variables_input_q ={'Variables': [T, rho, n , d],
#                   'Values': [3.924, 1.225, 4300/60, 0.3429],
#                   'Error_type': ['abs','abs','abs','abs'],
#                   'Error': [0.1, 0, 10/60 , 0.003]
#                   }

# Call the Taylor_error_propagation function with the expression and the dictionary as inputs
# The function performs a Taylor expansion of the first order on the given expression.
# It computes the derivative of the expression with respect to each variable, 
# calculates the uncertainties in the estimation of the desired quantity due to error in each of the individual variables,
# and finally computes the total uncertainty as the square root of the sum of the squares of the individual uncertainties.
# The function returns three outputs:
# 1. uncertainties: A list of the individual uncertainties for each variable.
# 2. sum_uncert: The total uncertainty, calculated as the square root of the sum of the squares of the individual uncertainties.
# 3. Expr_uncert: A symbolic expression for the total uncertainty.

uncertainties, sum_uncert, Expr_uncert, totValue= Taylor_error_propagation(T_Expr, dict_variables)

# Value, uncertMC = MonteCarlo_error_propagation(CT_Expr, dict_variables_input_q,1000)

print('Toqure Sensor')

print('Expected Torque \t\t{0}'.format(str(round(totValue,3))))
print('Expected Error  \t\t{0}'.format(str(round(sum_uncert,3))))
print('Error in % \t\t\t\t{0}'.format(str(round(sum_uncert/totValue*100,3))))

# Print the ratio of individual uncertainties to the total uncertainty
print(np.divide(uncertainties, sum_uncert))


