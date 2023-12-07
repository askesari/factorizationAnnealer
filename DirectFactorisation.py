"""
Direct Factorisation class
"""
from sympy import Pow, IndexedBase
import numpy as np
import math
# import custom modules
from logconfig import get_logger
from Factorisation import IntegerFactorisation

## initialize logger
log = get_logger(__name__)

class DirectFactorisation(IntegerFactorisation):
    def __init__(self, N):
        super().__init__(N)
        
        temp_expr = self.N - (self.p * self.q)
        self.direct_expr = temp_expr.expand()

        log.info(f"Direct Factorisation initialized with np = {self.np}, nq = {self.nq}")

    def get_expression(self):
        return(self.direct_expr)

    def get_norm_expression(self):
        """
        This method returns the squared norm of the expression.
        The squared norm is essential for the Hamiltonian formation.
        """
        # generate the squared norm of the expression in expanded form
        temp_expr = Pow(self.direct_expr,2).expand()

        # now, apply the simplification rule i.e. x**2 = x
        self.simplified_norm_expr = DirectFactorisation.apply_rule_simplify(temp_expr)

        return(self.simplified_norm_expr)
    
    @classmethod
    def get_terms_and_coeffs(cls, expr):
        """
        This method returns the individual terms and coefficients in the form of a dictionary 
        with the term as key and the coefficient as value.
        """
        # let us identify the terms and coefficients for each clause S_i
        clause_terms_list = list()
        clause_terms_list.append(expr.as_coefficients_dict())
        
        return(clause_terms_list)
    
    @classmethod
    def apply_rule_simplify(cls, expr):
        """
        Rule: This rule simplifies an expression e.g. x**2 = x
        """
        log.info(f"Apply Rule simplify begins ...")

        # first, let us get the individual terms and coefficients of the clauses
        clause_terms_list = DirectFactorisation.get_terms_and_coeffs(expr)
        
        # now, let us identify clauses satisfying the rule
        rule_keys = list()
        for clause_no, clause in enumerate(clause_terms_list):
            found_keys = list()
            for key, _ in clause.items():
                if key.func == Pow and key != 1:
                    found_keys.append(key)
                    
            if len(found_keys) > 0:
                rule_keys.extend(found_keys)
                log.info(f"Found keys: {found_keys} for clause {clause_no}")

        if len(rule_keys) == 0:
            log.info(f"Rule simplify: No keys found")

        # now, let us apply this rule to the input expression
        my_expr = expr
        for rule_key in rule_keys:
            my_expr = my_expr.subs({rule_key:rule_key.args[0]})
        
        log.debug(f"Rule simplify: {my_expr}")

        log.info(f"Rule simplify processing complete.")

        return(my_expr)