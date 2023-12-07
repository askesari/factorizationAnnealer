"""
This module executes the integer factorisation program using GAMA.
Author: Amit S. Kesari
"""
# import the basic and important modules 
import os
import numpy as np
import dimod, neal
from typing import Callable
import math
import itertools
import time
from sympy import IndexedBase

"""
from qiskit import QuantumCircuit, QuantumRegister, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver
from qiskit.visualization import plot_histogram
"""
# import custom modules
from Factorisation import IntegerFactorisation
from DirectFactorisation import DirectFactorisation
from ColumnFactorisation import ColumnFactorisation
from logconfig import get_logger
from Quadrization import Quadrization
from QUBOFormulation import QUBOFormulation
from GAMAFormulation import GAMAFormulation

## define some global variables
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
outputfilepath = curr_dir_path + "/output"

if not os.path.exists(outputfilepath):
    os.makedirs(outputfilepath)

## initialize logger
log = get_logger(__name__)

def is_folder_path_exists(folderpath):
    """
    Check if the folder path location exists and return True, if yes, otherwise False
    """
    ## initialize to False
    folder_exists = False

    try:
        if folderpath is not None:
            #v_file_dir = filename_with_path.rpartition("/")[0]
            try:
                if not os.path.exists(folderpath):
                    raise NameError("Folder path does not exist: " + folderpath)
            except NameError as ne:
                log.exception(ne, stack_info=True)
                raise
            else:
                folder_exists = True
        else:
            raise NameError("Folder path not passed as input: " + folderpath)          
    except NameError as ne1:
        log.exception(ne1, stack_info=True)
        raise
    
    return(folder_exists) 

# leveraging the function made available on the site
def get_solns_quantum(Q, offset, sampler, samples=20):
    """
    This function solves the QUBO and identifies the exact/optimal solutions.
    Note that suboptimal solutions are not considered in this implementation.
    """
    # Define Binary Quadratic Model
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q=Q, offset=offset)

    response = sampler.sample(bqm, num_reads=samples)

    response = response.aggregate()
    print(response)
    
    min_energy = min(response.record.energy)
    print(f"Minimum energy: {min_energy}")

    filter_idx = [i for i, e in enumerate(response.record.energy) if (e>=0 and e<=1)]

    optimal_sols = response.record.sample[filter_idx]

    return optimal_sols

def greedy(iterable):
  """
  Greedy approach for augmentation
  """
  for i, val in enumerate(iterable):
    if val[1] != 0:
      return i, val
  else:
    return i, val

def single_move(g: np.ndarray, fun: Callable, cost_func, x: np.ndarray, x_lo: np.ndarray = None, x_up: np.ndarray = None):
  """
  A single step move for augmentation (works well with greedy approach).
  This method has been updated to account for maximising the objective function.
  """
  if x_lo is None:
    x_lo = np.zeros_like(x)
  if x_up is None:
    x_up = np.ones_like(x)

  alpha = 0

  if (x + g <= x_up).all() and (x + g >= x_lo).all():
    if fun(x + g, cost_func) < fun(x, cost_func):
      alpha = 1
  elif (x - g <= x_up).all() and (x - g >= x_lo).all():
    if fun(x - g, cost_func) < fun(x, cost_func): #and fun(x - g) < fun(x + g):
      alpha = -1
  #print(f"g:{g},alpha:{alpha}, f(x-g):{fun(x-g)},f(x+g):{fun(x+g)},")
  return (fun(x + alpha*g, cost_func), alpha)

# Let us define the objective function
def f(x, cost_fn):
    """
    Evaluate the cost function for the 'x' values
    """
    x_sym = IndexedBase('x')
    for i in range(len(x)):
       cost_fn = cost_fn.subs(x_sym[i],x[i])
    return cost_fn

# Constraints definition
def constraint(x,A,b):
    return np.array_equiv(np.dot(A,x),b.T) or np.array_equiv(np.dot(A,x),b)

# Now, we define the augmentation process to find the minimum solution
def augmentation(basis, func, cost_func, x, A, b, VERBOSE: bool=True, itermax: int=1000):
  """
  This is the augmentation process
  """
  k = 0
  dist = 1
  if VERBOSE:
    print("Initial point:", x)
  iter_x = x
  x_lo = np.zeros_like(x)
  x_up = np.ones_like(x)
  while dist != 0 and k < itermax:
    k += 1
    g1, (obj, dist) = greedy(
              single_move(e, f, cost_func, iter_x, x_lo=x_lo, x_up = x_up) for e in basis)
    iter_x = iter_x + basis[g1]*dist
    if VERBOSE:
      print(f"Iteration: {k}")
      print(g1, (obj, dist))
      print(f"Augmentation direction: {basis[g1]}")
      print(f"Distanced moved: {dist}")
      print(f"Step taken: {basis[g1]*dist}")
      print(f"Objective function: {obj}")
      print(f"Current point: {iter_x}")
      print("Are constraints satisfied?:",constraint(iter_x, A, b))
    else:
      if k%10 == 0:
        print(f"Iteration: {k}")
        print(f"Objective function: {obj}")
        print(f"Current point: {iter_x}")
        print("Are constraints satisfied?:",constraint(iter_x, A, b))
  return(k,obj,x)


# start of main function
def main():
    log.info("=============================================")
    log.info(f"Start of program ...")
    log.info(f"Checking if output path exists ...")
    outputpath_exists = is_folder_path_exists(outputfilepath)

    # initialise the integer to be factorised
    N = 15
    
    # initialise column factorisation object
    log.info(f"Column Factorisation ... ")
    cf = ColumnFactorisation(N)
    binary_N = cf.get_binary_N()
    my_column_clauses = cf.get_column_clauses()
    my_p = cf.get_p()
    my_q = cf.get_q()
    log.info(f"Binary value of {N}: {binary_N}")
    log.info(f"p: {my_p}")
    log.info(f"q: {my_q}")
    
    for i, clause in enumerate(my_column_clauses):
        print(f"Column clause C{i+1}: {clause}")
    cf.classical_preprocessing(num_iterations = 15)
    my_column_clauses = cf.get_column_clauses()
    for i, clause in enumerate(my_column_clauses):
        print(f"Column clause C{i+1}: {clause}")

    cf_norm_expr = cf.get_norm_expression()
    cf_var_list = IntegerFactorisation.get_var_list(cf_norm_expr)
    log.info(f"Column Factorisation complete.")
    log.info(f"==============================================")
    log.info(f"Result ===>")
    log.info(f"N: {N}")
    log.info(f"Column: No. of variables: {len(cf_var_list)}; Variable list: {cf_var_list}") 
    log.info(f"Column Expression: {cf_norm_expr}")
    log.info(f"==============================================")
    

    # setup GAMA    
    q_gama = GAMAFormulation(cf_norm_expr, my_column_clauses)
    my_cost_func = q_gama.get_cost_function()
    my_constraint_expr = q_gama.get_constraint_expr()
    log.info(f"Constraints: {my_constraint_expr}")
    log.info(f"Cost func: {my_cost_func}")
    log.info(f"Var mapping: {q_gama.get_var_mapping()}")
    log.info(f"Column clauses: {my_column_clauses}")

    """
    solving via the GAMA approach
    """
    # getting kernel solutions
    A, B = q_gama.get_AxB_form()
    (rows_A,cols_A) = A.shape

    log.info(f"Ax=B: {A}, {B}")
    myQ_kernel, myQOffset_kernel = GAMAFormulation.get_Q_dict_and_offset(A)
    print(f"Q: {myQ_kernel}")
    print(f"Q Offset: {myQOffset_kernel}")

    simAnnSampler = neal.SimulatedAnnealingSampler()
    basisA = get_solns_quantum(myQ_kernel, myQOffset_kernel, sampler=simAnnSampler, samples = 500)

    # let us remove the 0 entry
    basisA = basisA[~np.all(basisA ==0, axis = 1)]

    print(len(basisA), ' kernel solutions found.')
    print("The kernel solutions are:")
    print(basisA)

    # getting feasible solutions
    Q_dict, Q_offset = GAMAFormulation.get_Q_dict_and_offset(A,B)
    log.info(f"Q-offset: {Q_offset}")
    log.info(f"Q-dict: {Q_dict}")
    

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q=Q_dict, offset=Q_offset)
    bqm.to_numpy_matrix()

    simAnnSampler = neal.SimulatedAnnealingSampler()
    feas_sols = get_solns_quantum(Q_dict, Q_offset, sampler=simAnnSampler, samples = 2000)
    log.info(f"{len(feas_sols)} feasible solutions found.")

    log.debug(f"The feasible solutions are:")
    for f_soln in feas_sols:
        log.debug(f_soln)

    init_obj = np.zeros((len(feas_sols),1))
    iters_full = np.zeros((len(feas_sols),1))
    final_obj_full = np.zeros((len(feas_sols),1))
    times_full = np.zeros((len(feas_sols),1))
    xf_full = np.empty((len(feas_sols),cols_A))
    for i,sol in enumerate(feas_sols):
        if not constraint(sol, A, B):
            log.info("Infeasible")
            pass
        init_obj[i] = f(sol, my_cost_func)
        start = time.process_time()
        iter, f_obj, xf = augmentation(basis = basisA, func = f, cost_func = my_cost_func, x = sol, A=A, b=B, VERBOSE=False)
        times_full[i] = time.process_time() - start
        iters_full[i] = iter
        final_obj_full[i] = f_obj
        xf_full[i] = xf
        log.debug(f"Sr.No {i+1}:The value of the objective function is: {f_obj} at the point: {xf}")
    # let us print the final solution
    log.info(f"***********************************")
    final_obj_value = min(final_obj_full)
    log.info(f"The min value of the objective function is: {final_obj_value}")
    final_obj_value_index = np.where(final_obj_full == final_obj_value)[0]
    log.info(f"The solution(s) providing the min value of objective function is/are: ")
    for i in final_obj_value_index:
        log.info(f"x = {xf_full[i][0:10]}, Slack x = {xf_full[i][10:15]}")

if __name__ == '__main__':
    main()    
