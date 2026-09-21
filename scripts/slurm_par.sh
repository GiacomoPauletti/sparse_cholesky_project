#!/bin/bash
#
# Submits one threaded profile_NS run (Cholesky, nested ordering) to SLURM.
# Not a batch script itself : run it directly,
#
#       ./slurm_par.sh <n> <threads> <out> [fact] [steps]
#
#       n       : sphere subdivision level
#       threads : OpenMP threads, and the CPUs reserved for the job
#       out     : report file
#       fact    : uplooking, multifrontal or parmultifrontal (default parmultifrontal)
#       steps   : time steps (default 1 : the factorization happens at setup)
#
# The binary is ./profile_NS unless PROFILE_NS points elsewhere.

set -euo pipefail

N=$1
THREADS=$2
OUT=$3
FACT=${4:-parmultifrontal}
STEPS=${5:-1}

case "$FACT" in
	uplooking | multifrontal | parmultifrontal) ;;
	*) echo "Unknown factorization '$FACT'." >&2; exit 1 ;;
esac

BIN=${PROFILE_NS:-./profile_NS}
if [ ! -x "$BIN" ]; then
	echo "profile_NS not found or not executable at '$BIN'." >&2
	exit 1
fi
BIN=$(readlink -f "$BIN")

# $N, $THREADS, ... are expanded now, at submit time; \$ survives to the job.
# srun gets --cpus-per-task explicitly : since Slurm 22.05 it no longer
# inherits it from sbatch, and every thread would land on a single core.
sbatch <<EOF
#!/bin/bash -l
#SBATCH -J chol_par
#SBATCH -o ./par_${FACT}_${N}_t${THREADS}.out
#SBATCH -e ./par_${FACT}_${N}_t${THREADS}.err
#SBATCH -D ./
#SBATCH --partition=p.tok.openmp
#SBATCH --qos=p.tok.openmp.2h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$THREADS
#SBATCH --mem=0
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun --cpus-per-task=$THREADS $BIN $N cholesky 1e-8 $STEPS $OUT nested $FACT
EOF
