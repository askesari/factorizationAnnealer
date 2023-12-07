"""
This module generates the QUBO based on GAMA method.
"""
from sympy import Pow, IndexedBase, srepr
from sympy.tensor.indexed import Indexed
import numpy as np
import math
# import custom modules
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class GAMAFormulation():
    def __init__(self, cost_func, sym_expr_list):
        
        self.var_list = list()
        self.var_mapping = list()

        if len(sym_expr_list) == 0:
            log.exception(f"Incorrect usage of GAMA Formulation. We need the column expressions to proceed.", stack_info=True)
            raise 

        # we use the GAMA format i.e. Ax=B as the constraints and 
        # norm expression as the cost function
        temp_cost_func = cost_func

        # define symbolic variables for linearization process
        self.y_sym = IndexedBase('y')
        self.pq_var_mapping = list()
        self.new_constraints = list()

        # we use the 'x' variables as the common base variables with which the p,q and y variables
        # will be replaced eventually.
        # To begin, we use these variables as part of creating the constraint equations.
        self.x_sym = IndexedBase('x')
        self.x_pos = 0

        # process the list of expressions
        for expr in sym_expr_list:
            # get the terms and coefficients within the expression
            #print(f"Expr: {expr}")
            terms_and_coeffs = expr.as_coefficients_dict()
            
            for term, _ in terms_and_coeffs.items():
                # check if any of the terms contain a product of 2 indexed variables
                temp_var_list = GAMAFormulation.get_free_symbols(term)
                #print(temp_var_list)
                if len(temp_var_list) == 2:
                    # need to apply the linearization process
                    v1 = temp_var_list[0]
                    v2 = temp_var_list[1]
                    new_var = self.y_sym[v1.args[1],v2.args[1]]
                    self.pq_var_mapping.append((term, new_var))
                    #print(srepr(term))
                    temp_cost_func = temp_cost_func.subs(term,new_var)

                    # include additional constraints
                    self.set_new_constraints(v1,v2,new_var)

        # now, that we added the new constraints and replaced the 2 term product by 
        # a single term, the cost function is automatically in the form of QUBO.
        # Now, we replace all p,q and y symbolic variables by a base variable 'x'

        ## But, first, let us identify the final list of symbols/variables used in the cost function.
        ## Since, the expresssion contains indexed symbols, we only include those in our list.
        ## e.g. p[1] is an indexed variable, whereas p is a symbol.
        temp_var_list = list(temp_cost_func.free_symbols)
        temp_var_list = sorted(temp_var_list,key=str)
        for v in temp_var_list:
            if v.func == Indexed:
                self.var_list.append(v)

        # replacing the existing variables by 'x' indexed variables
        my_expr = temp_cost_func
        for i in range(len(self.var_list)):
            v = self.var_list[i]
            my_expr = my_expr.subs(v,self.x_sym[self.x_pos])
            self.var_mapping.append((v,self.x_sym[self.x_pos]))
            self.x_pos += 1

        my_expr = my_expr.expand()
        log.info(f"The updated cost function consisting of only 'x' variables is: {my_expr}")
        # set the updated expression to object variable
        self.cost_func = my_expr

        # update the constraint variables p, q and y by the new 'x' variables
        temp_constraints = list()
        for constraint in self.new_constraints:
            for my_var in self.var_mapping:
                constraint = constraint.subs(my_var[0], my_var[1])
            temp_constraints.append(constraint)
        self.new_constraints = temp_constraints

        # we setup the Ax=B form i.e. identify matrix A and vector B
        self.set_AxB_form()

    @classmethod
    def get_free_symbols(cls, expr_term):
        """
        Returns the free indexed symbol list from a single term
        """
        my_var_list = list()
        #print(f"{expr_term}:{expr_term.free_symbols}")
        for v in sorted(expr_term.free_symbols, key=str):
            #print(f"{v}:{v.func}")
            if v!= 1 and v.func == Indexed:
                my_var_list.append(v)
        return my_var_list
    
    def set_new_constraints(self, p, q, y):
        """
        Convert a term of the form p[i]*q[j] into y[i,j] and include following 
        additional constraints:
        y[i,j] = p[i]*q[j]
        y[i,j] - p[i] + w[k] = 0
        y[i,j] - q[j] + w[k'] = 0
        p[i] + q[j] - y[i,j] + w[k''] = 1
        """
        # constraint 1: y[i,j] - p[i] + w[k]
        new_expr = y - p + self.x_sym[self.x_pos]
        self.x_pos += 1
        self.new_constraints.append(new_expr)

        # constraint 1: y[i,j] - q[j] + w[k']
        new_expr = y - q + self.x_sym[self.x_pos]
        self.x_pos += 1
        self.new_constraints.append(new_expr)

        # constraint 1: p[i] + q[j] - y[i,j] + w[k''] - 1
        new_expr = p + q - y + self.x_sym[self.x_pos] - 1
        self.x_pos += 1
        self.new_constraints.append(new_expr)

    def set_AxB_form(self):
        """
        This method sets the constraint equations in the form of matrix equation Ax=B
        Essentially, we need to identify the matrix A and vector B
        """
        num_rows = len(self.new_constraints)
        num_cols = self.x_pos # since we start 'x_pos' at 0
        self.matA = np.zeros((num_rows, num_cols))
        self.vecB = list()

        # process the constraints
        for row, constraint in enumerate(self.new_constraints):
            terms_and_coeffs = constraint.as_coefficients_dict()
            is_const_term_found = False
            for term, coeff in terms_and_coeffs.items():
                if term == 1:
                    self.vecB.append(-1*coeff)
                    is_const_term_found = True
                else:
                    col = term.args[1] 
                    self.matA[row,col] = coeff
            # if no constant term, then value is 0
            if is_const_term_found == False:
                self.vecB.append(0)

        # convert the list to numpy array / column vector
        self.vecB = np.array(self.vecB).reshape(num_rows,1)

    def get_AxB_form(self):
        """
        This method returns the matrix A and vector B that form Ax=B constraint equations
        """
        return(self.matA, self.vecB)

    @classmethod
    def get_Q_dict_and_offset(cls, A, b=None):
        """
        Setup the Q-dict and Q-offset value
        """
        # let us generate the Q-dict and offset value
        if b is None:
            Q = np.dot(A.T, A)
            Q_offset = 0.0
        else:
            AA = np.dot(A.T, A)
            h = -2.0*np.dot(b.T, A)
            Q = AA + np.diag(h[0])
            offset = np.dot(b.T, b) + 0.0
            Q_offset = offset[0,0]

        Q_dict = dict()
        # now, let us build the dictionary
        for row, rowVector in enumerate(Q):
            for col, coeff in enumerate(rowVector):
                Q_dict[row,col] = coeff

        return(Q_dict, Q_offset)
    
    def get_constraint_expr(self):
        """
        return the constraint expressions
        """
        return self.new_constraints

    def get_cost_function(self):
        """
        return the cost function to be evaluated
        """
        return self.cost_func
    
    def get_var_mapping(self):
        """
        return the mapping between 'x' variables and p,q,y and z variables
        """
        return self.var_mapping