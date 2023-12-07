"""
This module generates the QUBO from a single norm expression
"""
from sympy import Pow, IndexedBase
from sympy.tensor.indexed import Indexed
import numpy as np
import math
# import custom modules
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class QUBOFormulation():
    def __init__(self, cost_func):
        
        self.Q_dict = dict()
        self.Q_offset = 0.0
        self.var_list = list()
        self.var_mapping = list()
         
        ## let us identify the symbols/variables used in the expression
        ## since, the expresssion contains indexed symbols, we only include those in our list
        ## e.g. p[1] is an indexed variable, whereas p is a symbol.
        temp_var_list = list(cost_func.free_symbols)
        temp_var_list = sorted(temp_var_list,key=str)
        for v in temp_var_list:
            if v.func == Indexed:
                self.var_list.append(v)

        # now, let us replace all the different base variables by a single base variable
        # this will ensure that we have the appropriate sequence of indexed variables to
        # form the Q dictionary
        x_sym = IndexedBase('x')
        my_expr = cost_func
        for i in range(len(self.var_list)):
            v = self.var_list[i]
            my_expr = my_expr.subs({v:x_sym[i]})
            self.var_mapping.append((v,x_sym[i]))
        my_expr = my_expr.expand()
        log.info(f"The updated expression consisting of only 'x' variables is: {my_expr}")
        # set the updated expression to object variable
        self.cost_func = my_expr

        # now, we identify the Q dictionary and the Q offset value
        self.set_Q_dict_and_offset()
   

    def set_Q_dict_and_offset(self):
        """
        Setup the Q-dict and Q-offset value
        """

        # step 1: let us get the individual terms
        terms = self.cost_func.as_coefficients_dict()

        # step 2: identify x_i term, x_i*x_j term and constant term
        for term, coeff in terms.items():
            if term == 1:
                # this is the offset value
                self.Q_offset = coeff
            elif term.args[0].func == IndexedBase: # s_i term found
                my_i = term.args[1]
                self.Q_dict[(my_i,my_i)] = coeff
            elif term.args[0].func == Indexed: # s_i*s_j term found
                my_i = term.args[0].args[1]
                my_j = term.args[1].args[1]
                if my_i > my_j:
                    # swap the elements
                    temp_i = my_i
                    my_i = my_j
                    my_j = temp_i
                self.Q_dict[(my_i,my_j)] = coeff
        log.info("Q-dict and Q-offset setup for NORM expression.")

    def get_Q_dict_and_offset(self):
        """
        Returns the Q-dict and Q-offset value
        """
        return(self.Q_dict, self.Q_offset)
    
    def get_var_mapping(self):
        """
        Returns the mapping between the old variable and the new 'x' variable
        """
        return(self.var_mapping)