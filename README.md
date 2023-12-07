# Bi-Prime Integer Factorization using Quantum Annealer

## Description
This project is part of "Quantum Integer Programming" course (ID5840) conducted at Indian Institute of Technology - Madras. The objective of the project is to leverage binary optimization to factorize bi-prime integers using either Simulated Annealing or D-Wave QPU.

Two methods have been explored as part of this project:
1. Direct factorisation method
2. Column-based multiplication method

Additionally, the column-based multiplication method can be utlized in 2 distinct approaches: 
1. Quadratization that converts higher order terms i.e. $3$-local and $4$-local terms to quadratic
2. Graver Augmented Multi-seed Algorithm (GAMA)

## Pre-requisites
The pre-requisites for installing the package are:

### Python==3.8.13
It is advisable to create a new envionment using either pip or conda to deploy the project. 
If using conda, following command can be used where \<envname> needs to be replaced with appropriate name during execution. 
    
    conda create --name <envname> python==3.8.13 

### D-Wave Ocean SDK
The following command can be used to install D-Wave Ocean SDK APIs

    pip install dwave-ocean-sdk

> Note: Installation and execution of the package has been tested on both unix/linux and windows 10 operating systems.

> Note: When running the program, it is important to choose a value of N that has factors P & Q with approximately equal strength i.e. similar no. of bits. e.g. N=143, 899, etc.

## Usage
1. Direct method via Simulated Annealing: Update the value of N in the file run_df_integer_factorisation.py and execute it.

        python3 run_df_integer_factorisation.py
   
2. Column-based method via Simulated Annealing: Update the value of N in the file run_cf_integer_factorisation.py and execute it.

       python3 run_cf_integer_factorisation.py
   
3. Column-based method via D-Wave QPU: Update the value of N in the file run_dwave_cf_integer_factorisation.py and execute it.

       python3 run_dwave_cf_integer_factorisation.py
   
4. Column-based method via GAMA: Update the value of N in the file run_gama_int_factorisation.py and execute it.

       python3 run_gama_int_factorisation.py

## Author(s)

Amit Shashikant Kesari

## License
[Apache2.0](https://opensource.org/licenses/Apache-2.0)
