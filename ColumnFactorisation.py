"""
Column Based Factorisation class
"""
from sympy import Pow, IndexedBase
from sympy.tensor.indexed import Indexed
import numpy as np
import math
# import custom modules
from logconfig import get_logger
from Factorisation import IntegerFactorisation

## initialize logger
log = get_logger(__name__)

class ColumnFactorisation(IntegerFactorisation):
    def __init__(self, N):
        super().__init__(N)

        # define the carry variables z[i,j]
        # IMP: these z variables also need to be included in the self.var_list
        self.z_sym = IndexedBase('z')

        # set up individual column as a clause
        self.C_list = list()
        # also, let us define a dictionary to hold the values of the variables obtained via
        # classical preprocessing
        self.var_values_dict = dict()

        # we define a matrix M to indicate the position of carry terms generated
        matM = np.zeros((self.np + self.nq + 1, self.np + self.nq + 1))
        
        # also, we need to have a reverse copy of self.binary_N to ensure that
        # the correct clauses have the appropriate bit
        reverse_binary_N = self.binary_N.copy()
        reverse_binary_N.reverse()

        for i in range(1, self.np + self.nq + 1):
            column_expr = 0
            # define the p*q terms
            # the starting point is li=max(i-np+1,0) assuming np>=nq 
            l = max(i - self.np + 1, 0)
            nl = 0 # number of p*q terms
            for j in range(l, min(self.nq,i+1)):
                column_expr = column_expr - self.q_sym[j] * self.p_sym[i-j]
                nl = nl + 1

            # define the input carry terms 
            # we know C_1 clause does not have an input carry.
            # so, we only consider the input carry from C_2 clause
            # also, we leverage the matrix M (matM) used for identifying the output carries
            # from previous clauses to determine the input carries for the subsequent clauses
            nz = 0 # number of input carry terms
            if i > 1:
                for j in range(1, i):
                    if matM[j][i] == 1:
                        column_expr = column_expr - self.z_sym[j,i]
                        nz = nz + 1
                   
            # define the output carry terms
            # we leverage the matrix M defined outside the loop for identifying the generated 
            # output carry positions
            if (nl + nz) > 0: # if no input terms, then output is also zero
                if i>1:
                    # the starting point is j=1, but end point is mi = ceil(log2(nl + nz))
                    temp_m = math.log2(nl + nz)
                    if temp_m.is_integer():
                        m = int(temp_m) + 1
                    else:
                        m = math.ceil(temp_m)

                    for j in range(1, m):
                        column_expr = column_expr + Pow(2,j) * self.z_sym[i,i+j]
                        matM[i][i+j] = 1
                        # include the carry term in the variable list self.var_list
                        #self.var_list[self.z_sym[i,i+j]] = {'ising':1}
                else: # a carry can be generated from S_1 clause
                    # we manually set the matM[1][2] value as 1
                    column_expr = column_expr + Pow(2,i) * self.z_sym[i,i+1]
                    matM[i][i+1] = 1
                    # include the carry term in the variable list self.var_list
                    #self.var_list[self.z_sym[i,i+1]] = {'ising':1}

            # finally define the output Ni term i.e. bit in ith position of N
            if(i<self.nm):
                column_expr = column_expr + reverse_binary_N[i]

            # now, set p0=q0=1
            if column_expr != 0:
                column_expr = column_expr.subs({self.p_sym[0]:1,self.q_sym[0]:1})

                self.C_list.append(column_expr)

        log.info(f"Column Based Factorisation initialized with np = {self.np}, nq = {self.nq} and Ci clauses = {len(self.C_list)}")

    def get_column_clauses(self):
        return(self.C_list)
    
    def get_var_values_dict(self):
        """
        Returns the values/expressions set for the variables as part of classical pre-processing
        """
        return self.var_values_dict
    
    def get_terms_and_coeffs(self):
        """
        This method breaks down each clause C_i into individual terms and coefficients and 
        returns in the form of a dictionary with the term as key and the coefficient as value.
        """
        # let us identify the terms and coefficients for each clause S_i
        clause_terms_list = list()
        for clause in self.C_list:
            clause_terms_list.append(clause.as_coefficients_dict())
        
        return(clause_terms_list)

    @classmethod
    def apply_rule_1(cls, terms_dict):
        """
        Rule: x1+x2+...+xn = n => x_i = 1 
           or -x1-x2-...-xn = -n => x_i = 1
        """
        log.info(f"Apply Rule 1 begins for {terms_dict} ...")

        # now, identify terms satisfying the rule
        rule_keys = list()
        # initialise the appropriate values
        positive_coeff = 0
        negative_coeff = 0
        const_coeff = 0
        is_found = True
        found_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                const_coeff = const_coeff + coeff
            elif coeff == -1:
                found_keys.append(key)
                negative_coeff = negative_coeff + abs(coeff)
            elif coeff == 1:
                found_keys.append(key)
                positive_coeff = positive_coeff + coeff
            else:
                is_found = False
                break

        if len(found_keys) > 0:
            if (is_found == True and positive_coeff == 0 and negative_coeff == const_coeff) or \
                (is_found == True and negative_coeff == 0 and positive_coeff == (-1)*const_coeff):
                rule_keys = found_keys
                log.info(f"Rule 1: Found keys {found_keys}")

        return(rule_keys)

    @classmethod
    def apply_rule_2(cls, terms_dict):
        """
        Rule: x1 + x2 + x3 + ... = 0 => all x_i = 0 OR
              -x1 -x2 -x3 -... = 0 => all x_i = 0 
        """
        log.info(f"Apply Rule 2 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise the appropriate values    
        positive_coeff = 0
        negative_coeff = 0
        const_coeff = 0
        is_found = True
        found_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                const_coeff = const_coeff + coeff
            elif coeff == -1:
                negative_coeff = negative_coeff + abs(coeff)
                found_keys.append(key)
            elif coeff == 1:
                positive_coeff = positive_coeff + coeff
                found_keys.append(key)
            else:
                is_found = False
                break

        if (is_found == True and negative_coeff > 0 and positive_coeff == 0 and const_coeff == 0) or \
            (is_found == True and positive_coeff > 0 and negative_coeff == 0 and const_coeff == 0):
            rule_keys = found_keys
            log.info(f"Rule 2: Found keys {found_keys}")

        return(rule_keys)

    @classmethod
    def apply_rule_3(cls, terms_dict):
        """
        Rule: x1 + x2 = 2*x3 => all x1=x2=x3
        """
        log.info(f"Apply Rule 3 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        if len(terms_dict) == 3:
            # initialise appropriate values
            positive_coeff = 0
            positive_coeff_terms = 0 
            negative_coeff = 0
            negative_coeff_terms = 0
            is_found = True
            found_keys = list()
            for key, coeff in terms_dict.items():
                if key == 1:
                    is_found = False
                    break
                elif coeff < 0:
                    negative_coeff = negative_coeff + abs(coeff)
                    negative_coeff_terms += 1
                    found_keys.append(key)
                else:
                    positive_coeff = positive_coeff + coeff
                    positive_coeff_terms += 1
                    found_keys.append(key)

            if (is_found == True) and (negative_coeff == positive_coeff) and \
                (positive_coeff_terms == 1 or negative_coeff_terms == 1) and len(found_keys) != 0:
                rule_keys = found_keys
                log.info(f"Rule 3: Found keys {found_keys}")

        return(rule_keys)

    @classmethod
    def apply_rule_4(cls, terms_dict):
        """
        Rule: -x1-x2-...-xn + c1*y1 + c2*y2 + ... cn*yn + c0, c_i's are constants 
                        => if c_i > abs(negative_coeff) - c0, then y_i = 0
              A specific case: x1 + x2 = 2*x3 + 1 => x3 = 0
        """
        log.info(f"Apply Rule 4 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise appropriate values
        const_coeff = 0
        negative_coeff = 0
        found_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                const_coeff = const_coeff + coeff
            elif coeff < 0:
                negative_coeff = negative_coeff + abs(coeff)
        # compute the value to verify. 
        # Here, negative_coeff has to be greater than const_coeff; otherwise it is an invalid condition
        check_coeff = negative_coeff - const_coeff

        for key, coeff in terms_dict.items():
            if key != 1 and coeff > check_coeff:
                found_keys.append(key)

        if len(found_keys) > 0:
            rule_keys = found_keys
            log.info(f"Rule 4: Found keys {found_keys}")

        return(rule_keys)    

    @classmethod
    def apply_rule_5(cls, terms_dict):
        """
        Rule: x1 + x2 = 1 => x1.x2 = 0
        """
        log.info(f"Apply Rule 5 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise appropriate values
        if len(terms_dict) == 3:
            const_coeff = 0
            found_coeff = 0
            is_found = True
            found_keys = list()
            for key, coeff in terms_dict.items():
                if key == 1:
                    const_coeff = const_coeff + coeff
                elif coeff == -1:
                    found_coeff = found_coeff + abs(coeff)
                    found_keys.append(key)
                else:
                    is_found = False
                    break

            if (is_found == True) and (found_coeff == 2) and (const_coeff == 1):
                my_key = 1
                for key in found_keys:
                    my_key = my_key * key
                rule_keys = [my_key]
                log.info(f"Rule 5: Found keys {found_keys}")

        return(rule_keys)

    
    @classmethod
    def apply_rule_simplify(cls, terms_dict):
        """
        Rule: This rule simplifies an expression e.g. x**2 = x
        """
        log.info(f"Apply Rule simplify begins for {terms_dict} ...")

        # now, let us check if the expression satsifies the rule
        rule_keys = list()
        for term, coeff in terms_dict.items():
            if term.func == Pow and term != 1:
                rule_keys.append(term)
                    
        if len(rule_keys) > 0:
                log.info(f"Found keys: {rule_keys}")

        return(rule_keys)
    
    def classical_preprocessing(self, num_iterations):
        """
        This method applies classical pre-processing rules to the individual columns so as to reduce the number of variables and hence the number of qubits.
        num_iterations: There can be multiple passes over the clauses using the same rules to ensure that rules applied in previous iteration can reduce the expression variables in the subsequent passes.
        """
        
        for iter in range(num_iterations):
            log.info(f"Pass {iter+1} of the classical processing rules ... ")

            ### apply rule 1 i.e. (x1+x2+..+xn = n or -x1-x2-...-xn = -n) => x_i = 1
            # first, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_1(terms_dict=clause_terms))
                
            # now, let us apply this rule to all the clauses
            if len(rule_keys) == 0 :
                log.info(f"Rule 1: No new keys found")
            else:
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        self.C_list[i] = self.C_list[i].subs({rule_key:1})
                    self.var_values_dict[rule_key] = 1
                log.info(f"Rule 1 processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Clause {s_no+1}: {clause}")

            ### apply rule 2 i.e. (x1+x2+x3+...=0 or -x1-x2-x3-..=0) => x_i = 0 
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_2(terms_dict=clause_terms))
                
            # now, let us apply this rule to all the clauses
            if len(rule_keys) == 0 :
                log.info(f"Rule 2: No new keys found")
            else:
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        self.C_list[i] = self.C_list[i].subs({rule_key:0})
                    self.var_values_dict[rule_key] = 0
                log.info(f"Rule 2 processing complete.")
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Clause {s_no+1}: {clause}")
                

            ### apply rule 3 i.e. (x1+x2+x3+...+xn = n.y or -x1-x2-x3-..-xn = -n.y) => 
            ###                                                   x_1 = x_2 = ... = y 
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                found_keys = ColumnFactorisation.apply_rule_3(terms_dict=clause_terms)
                # ensure that empty lists are not added to rule_keys list
                if len(found_keys) > 0:
                    rule_keys.append(found_keys)
                
            if len(rule_keys) == 0:
                log.info(f"Rule 3: No new keys found.")
            else:
                # now, let us apply this rule to all the clauses
                print(rule_keys)
                for rule_key in rule_keys:
                    print(rule_key)
                    main_key = rule_key[0] # this main key will replace all the other keys that carry   
                                           # the same value.
                    for i in range(len(self.C_list)):
                        for key_indx in range(1, len(rule_key)):
                            self.C_list[i] = self.C_list[i].subs({rule_key[key_indx]:main_key})
                    # since each rule key contains 3 terms, we add them to the var_values_dict
                    # in pairs, so that we know which p, q and/or z variables are equivalent
                    self.var_values_dict[rule_key[1]] = main_key
                    self.var_values_dict[rule_key[2]] = main_key

            log.info(f"Rule 3 processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 3: Clause {s_no+1}: {clause}")

            #### apply rule 4
            # -x1-x2-...-xn + c1*y1 + c2*y2 + ... cn*yn + c0, c_i's are constants 
            # => if c_i > abs(negative_coeff) - c0, then y_i = 0
            # a specific case: (x+y=2z+1 => z=0)
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_4(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule 4: No new keys found.")

            # now, let us apply this rule to all the clauses
            for rule_key in rule_keys:
                for i in range(len(self.C_list)):
                    self.C_list[i] = self.C_list[i].subs({rule_key:0})
                self.var_values_dict[rule_key] = 0

            log.info(f"Rule 4 processing complete.")
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 4: Clause {s_no+1}: {clause}")

            #### apply rule 5 i.e. (x1+x2=1 => x1.x2=0)
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_5(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule 5: No new keys found.")

            # now, let us apply this rule to all the clauses
            for rule_key in rule_keys:
                for i in range(len(self.C_list)):
                    self.C_list[i] = self.C_list[i].subs({rule_key:0})
                self.var_values_dict[rule_key] = 0

            log.info(f"Rule 5 processing complete.")
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 5: Clause {s_no+1}: {clause}")


            ### apply rule simplify i.e. x**2 = x 
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_simplify(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule simplify: No keys found")

            # now, let us apply this rule to all the clauses    
            for rule_key in rule_keys:
                for i in range(len(self.C_list)):
                    self.C_list[i] = self.C_list[i].subs({rule_key:rule_key.args[0]})
        
            log.info(f"Rule simplify processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule simplify: Clause {s_no+1}: {clause}")


    def get_norm_expression(self):
        """
        This method returns the simplified squared norm of the expression.
        The squared norm is essential for the Hamiltonian formation.
        """
        log.info(f"Generating norm expression ...")
        
        # generate the squared value of the expression in expanded form
        # here, we square each individual clause and finally add them all together
        temp_expr = None
        for expr in self.C_list:
            if temp_expr is None:
                temp_expr = Pow(expr,2).expand()
            else:
                temp_expr = temp_expr + Pow(expr,2).expand()
        
        # now, we simplify the expression based on the simplification rule i.e. x**2=x1.
        # IMP: It is important to not apply any other simplification rules as they have already
        #      been covered in the individual column expressions. Adding the other rules, may also
        #      result in the Ising Hamiltonian having eigen value < 0.
        # first, let us get the terms and coefficients of the norm expression
        terms_and_coeff = temp_expr.as_coefficients_dict()
        
        rule_keys = ColumnFactorisation.apply_rule_simplify(terms_and_coeff)
        if len(rule_keys) == 0:
            log.info(f"Rule simplify: No keys found")

        # now, let us apply this rule to all the clauses    
        for rule_key in rule_keys:
            temp_expr = temp_expr.subs({rule_key:rule_key.args[0]})

        log.debug(f"After rule - simplify: {temp_expr}")
        log.info(f"Norm expression processing complete.")

        self.simplified_norm_expr = temp_expr
        
        return(self.simplified_norm_expr)