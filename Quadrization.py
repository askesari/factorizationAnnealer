### Author: Aravind, Amit

from sympy import *
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

def Quadrization(expression):
    dic = expression.as_coefficients_dict()
    pos = 0
    neg = 0
    final_expression = 0
    for term in dic:
        if len(term.args_cnc()[0]) > 2:
            if dic[term] > 0:
                pos += (dic[term]*term)
            else:
                neg += (dic[term]*term)
        else:
            final_expression += (dic[term]*term)
    # print(final_expression)
    # print(pos)
    # print(neg)
    y = IndexedBase('y')
    final_pos = 0
    new_qubits = 1
    if pos != 0:
        dic = pos.as_coefficients_dict()
    else:
        dic = {}
    replaced_dic = {}
    mapping = {}
    for term in dic:
        rl = len(term.args_cnc()[0])-2    # Recursion Length
        curr_term = term
        for iteration in range (rl):
            new_term = 1
            li = curr_term.args_cnc()[0]
            for j in range (rl+2-iteration):
                if j < (rl+1-iteration):
                    new_term *= li[j]
            # final_pos += (li[rl+1-iteration]*y[new_qubits])

            replaced_dic[y[new_qubits]] = li[rl+1-iteration]
            if li[rl+1-iteration] in mapping:
                mapping[li[rl+1-iteration]].append(y[new_qubits])
            else:
                mapping[li[rl+1-iteration]] = [y[new_qubits]]

            curr_term = new_term
            neg += (-1 * dic[term]* new_term * (y[new_qubits]))
            new_qubits += 1
        final_pos += (dic[term]*curr_term)    
    # print(final_pos)

    final_neg = 0
    if neg != 0:
        dic = neg.as_coefficients_dict()
    else:
        dic = {}
    for term in dic:
        final_term = 0
        l = len(term.args_cnc()[0])  # Number of elements in term
        for j in term.args_cnc()[0]:
            final_term += j
        final_term -= (l-1)
        final_term *= y[new_qubits]
        new_qubits += 1
        final_term *= dic[term]
        final_term = expand(final_term)
        final_neg += final_term

    fterm = final_expression + final_pos + final_neg
    dic = fterm.as_coefficients_dict()
    fterm1 = 0
    for term in dic:
        term1 = 1
        l = term.args_cnc()[0]
        for i in l:
            if i in replaced_dic:
                term1 *= mapping[replaced_dic[i]][0]
            else:
                term1 *= i
        fterm1 += dic[term]*term1

    # let us find the weighting factor w i.e. sum of coefficients + some constant
    w = 0
    terms_and_coeffs = fterm1.as_coefficients_dict()
    for term, coeff in terms_and_coeffs.items():
        w = w + coeff
    w = abs(w)+1
    log.debug(f"Weighting factor: {w}")
    
    for i in mapping:
        fterm1 += w*(1 - i - mapping[i][0] + (2*i*mapping[i][0]))

    # print(final_neg)
    return (fterm1)
