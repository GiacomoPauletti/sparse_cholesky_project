import os

for n_exp in range(4,7):
    n= 2**n_exp
    for t in [100, 200, 400, 800, 1600]:
        for tol in [1e-08, 1e-10, 1e-12]:
           os.system(f"./slurm_seq.sh {n} cg {tol} {t}")
        os.system(f"./slurm_seq.sh {n} cholesky {tol} {t}")
