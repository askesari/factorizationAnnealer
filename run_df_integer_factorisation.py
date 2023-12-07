"""
This module executes the integer factorisation program for direct method.
Author: Amit S. Kesari
"""
# import the basic and important modules 
import os
import numpy as np
import dimod, neal
import math
from dwave.preprocessing import roof_duality
from greedy import SteepestDescentSolver

# import custom modules
from Factorisation import IntegerFactorisation
from DirectFactorisation import DirectFactorisation
from ColumnFactorisation import ColumnFactorisation
from logconfig import get_logger
from Quadrization import Quadrization
from Quadrization2 import Quadrization2
from QUBOFormulation import QUBOFormulation
from sympy import IndexedBase

## define some global variables
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
outputfilepath = curr_dir_path + "/output"

if not os.path.exists(outputfilepath):
    os.makedirs(outputfilepath)

simAnnSampler = neal.SimulatedAnnealingSampler()
solver_greedy = SteepestDescentSolver()

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
# def get_solns_quantum(Q, offset, sampler, samples=20, init_state=None):
#     """
#     This function solves the QUBO and identifies the exact/optimal solutions.
#     Note that suboptimal solutions are not considered in this implementation.
#     """
#     # Define Binary Quadratic Model
#     bqm = dimod.BinaryQuadraticModel.from_qubo(Q=Q, offset=offset)

#     if init_state is None:
#         response = sampler.sample(bqm, num_reads=samples)
#     else:
#         response = sampler.sample(bqm, num_reads=len(init_state), initial_states=init_state)


#     response = response.aggregate()
#     #print(response)
    
#     min_energy = min(response.record.energy)
#     log.info(f"Minimum energy found: {min_energy}")

#     filter_idx = [i for i, e in enumerate(response.record.energy) if e == min_energy]

#     optimal_sols = response.record.sample[filter_idx]

#     return optimal_sols

def get_solns_quantum(Q, offset, sampler, samples=20, init_state=None):
    """
    This function solves the QUBO and identifies the exact/optimal solutions.
    Note that suboptimal solutions are not considered in this implementation.
    """
    # Define Binary Quadratic Model
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q=Q, offset=offset)

    # check if preprocessing helps
    preprocess_sols = roof_duality(bqm)
    log.info(f"Preprocessing results: {preprocess_sols}")

    if init_state is None:
        log.info(f"No initial state. num_reads = {samples}.")
        init_response = sampler.sample(bqm, num_reads=samples)
    else:
        log.info(f"Initial state set. Using num_reads accordingly.")
        init_response = sampler.sample(bqm, num_reads=len(init_state), initial_states=init_state)

    init_min_energy = min(init_response.aggregate().record.energy)
    log.info(f"Minimum energy found in initial response: {init_min_energy}")

    log.info("Postprocessing initial results with greedy solver ... ")
    response = solver_greedy.sample(bqm, initial_states=init_response)

    response = response.aggregate()
    #print(response)
    
    min_energy = min(response.record.energy)
    log.info(f"Minimum energy found after postprocessing: {min_energy}")

    filter_idx = [i for i, e in enumerate(response.record.energy) if e == min_energy]

    optimal_sols = response.record.sample[filter_idx]

    return optimal_sols

# start of main function
def main():
    log.info("=============================================")
    log.info(f"Start of program ...")
    log.info(f"Checking if output path exists ...")
    outputpath_exists = is_folder_path_exists(outputfilepath)

    # initialise the integer to be factorised
    N = 3233
    
    # initialise direct factorisation object
    df = DirectFactorisation(N)
    binary_N = df.get_binary_N()
    my_expr = df.get_expression()
    my_p = df.get_p()
    my_q = df.get_q()
    df_norm_expr = df.get_norm_expression()
    log.info(f"Direct Factorisation ... ")
    log.info(f"Binary value of {N}: {binary_N}")
    log.info(f"p: {my_p}")
    log.info(f"q: {my_q}")
    log.info(f"N-p*q expression: {my_expr}")
    log.info(f"(N-p*q)^2 expression: {df_norm_expr}")
    df_var_list = IntegerFactorisation.get_var_list(df_norm_expr)
    log.info(f"Variable list: {df_var_list}")
    log.info(f"Direct Factorisation complete.")

    log.info(f"==============================================")
    log.info(f"Result ===>")
    log.info(f"N: {N}")
    log.info(f"Direct: No. of variables: {len(df_var_list)}; Variable list: {df_var_list}")
    log.info(f"Direct Expression: {df_norm_expr}")
    log.info(f"==============================================")

    
    #Quadrization Part
    df_qubo = Quadrization2(df_norm_expr)
    total_num_vars = len(IntegerFactorisation.get_var_list(df_qubo))
    print("====================================================")
    print(f"Direct: No. of variables after quadratization: {total_num_vars}")
    print(df_qubo)
       
    # setup QUBO
    q_qubo = QUBOFormulation(df_qubo)
    Q_dict, Q_offset = q_qubo.get_Q_dict_and_offset()
    log.info(f"Q_Offset: {Q_offset}, Q_dict: {Q_dict}")

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q=Q_dict, offset=Q_offset)
    log.debug(f"{bqm.to_numpy_matrix()}")

    simAnnSampler = neal.SimulatedAnnealingSampler()
    # if initial state is passed, the sample size is the same as the number of possible initial states
    # otherwise, the number of samples passed as input is considered
    bqm_initial_state = [1]*total_num_vars
    feas_sols = get_solns_quantum(Q_dict, Q_offset, sampler=simAnnSampler, samples = 500, init_state=None)
    log.info(f"{len(feas_sols)} feasible solutions found.")

    log.info(f"The feasible solutions are:")
    for f_soln in feas_sols:
        log.info(f_soln)
    log.info(f"Variable mapping is: {q_qubo.get_var_mapping()}")

    """
    let us now work with the feasible solutions and identify the decimal output
    """
    # step 1: let us find 'x' variable indexes corresponding to 'p' and 'q' variables
    pq_x_map_list = list()
    p_found_list = list()
    q_found_list = list()
    for var_map in q_qubo.get_var_mapping():
        if var_map[0].args[0] == IndexedBase('p'):
            p_found_list.append(var_map[0].args[1])
            pq_x_map_list.append(var_map)
        if var_map[0].args[0] == IndexedBase('q'):
            q_found_list.append(var_map[0].args[1])
            pq_x_map_list.append(var_map)
    log.debug(pq_x_map_list)

    # step 2: let us navigate through the feasible solns. to find the values
    f_pq_soln = set()
    for f_soln in feas_sols:
        temp_pq_soln = list()
        for pq_x_map in pq_x_map_list:
            x_ind = pq_x_map[1].args[1]
            temp_pq_soln.append((pq_x_map[0],f_soln[x_ind]))
        temp_pq_soln = tuple(temp_pq_soln)
        f_pq_soln.add(temp_pq_soln)
    log.debug(f"PQ Soln pair: {f_pq_soln}")
    
    # step 3: get the decimal solution
    final_soln = list()
    for soln in f_pq_soln:
        num_p = 1
        num_q = 1
        for pq_val in soln:
            if pq_val[0].args[0] == IndexedBase('p'):
                num_p = num_p + math.pow(2,pq_val[0].args[1]) * pq_val[1]
            elif pq_val[0].args[0] == IndexedBase('q'):
                num_q = num_q + math.pow(2,pq_val[0].args[1]) * pq_val[1]
        final_soln.append((num_p,num_q))
    log.info(f"Final Soln: {final_soln}")
    var_dict = IntegerFactorisation.get_var_components_dict(df_qubo)
    for key, value in var_dict.items():
        log.info(f"Num of variables after QUBO for {key}: {len(value)}")

if __name__ == '__main__':
    main()    
