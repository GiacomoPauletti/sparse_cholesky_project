#!/bin/bash
#
# Submits one sequential profile_NS run to SLURM.
#
# This script is NOT itself a batch script : run it directly,
#
#       ./slurm_seq.sh [n] [cholesky|cg] [tol] [steps] [out] [natural|nested] \
#                      [uplooking|multifrontal|parmultifrontal]
#
# and it submits the job below through sbatch. The batch script is the
# heredoc at the bottom, fed to sbatch on its standard input, so there is no
# second file to keep in sync.
#
# Arguments are those of profile_NS and default the same way :
#
#       n      : sphere subdivision level     (default 16)
#       solver : cholesky or cg               (default cholesky)
#       tol    : CG relative residual target  (default 1e-8)
#       steps  : number of time steps         (default 100)
#       out    : report file                  (default performance.txt)
#       order  : vertex ordering, Cholesky only (default natural)
#       fact   : factorization, Cholesky only  (default uplooking)
#
# Give distinct out names when submitting a sweep : all the jobs share the
# submit directory, so they would otherwise overwrite each other's report.
#
# The binary is ./profile_NS by default ; override it with the PROFILE_NS
# environment variable, e.g.
#
#       PROFILE_NS=~/sparse_cholesky_project/build/release/profile_NS \
#               ./slurm_seq.sh 32 cg 1e-10
#
# The job runs in the directory sbatch was called from (#SBATCH -D ./), which
# is where profile_NS writes its performance.txt, and where the .out/.err
# files land.

set -euo pipefail

N=${1:-16}
SOLVER=${2:-cholesky}
TOL=${3:-1e-8}
STEPS=${4:-100}
OUT=${5:-performance.txt}
ORDER=${6:-natural}
FACT=${7:-uplooking}

case "$SOLVER" in
	cholesky | cg) ;;
	*)
		echo "Unknown solver '$SOLVER' (expected cholesky or cg)." >&2
		exit 1
		;;
esac

case "$ORDER" in
	natural | nested) ;;
	*)
		echo "Unknown ordering '$ORDER' (expected natural or nested)." >&2
		exit 1
		;;
esac

case "$FACT" in
	uplooking | multifrontal | parmultifrontal) ;;
	*)
		echo "Unknown factorization '$FACT' (expected uplooking," \
			"multifrontal or parmultifrontal)." >&2
		exit 1
		;;
esac

# Resolved to an absolute path now, at submit time : the compute node runs the
# job in the submit directory, but spelling the binary out avoids any surprise
# if that ever changes.
BIN=${PROFILE_NS:-./profile_NS}
if [ ! -x "$BIN" ]; then
	echo "profile_NS not found or not executable at '$BIN'." >&2
	echo "Build it first, or point PROFILE_NS at it." >&2
	exit 1
fi
BIN=$(readlink -f "$BIN")

# Inside the heredoc, the shell expands $N, $SOLVER, ... before sbatch sees
# them, which is what makes the #SBATCH directives parameterizable at all :
# they are comments, so sbatch would never expand them itself. Anything that
# must survive until the job actually runs is escaped (\$SLURM_CPUS_PER_TASK).
if [ "$SOLVER" == "cholesky" ]; then
sbatch <<EOF
#!/bin/bash -l
#SBATCH -J $SOLVER            #Job name
#SBATCH -o ./${SOLVER}_${ORDER}_${FACT}_${N}_${STEPS}.out        #stdout (%x=jobname, %j=jobid)
#SBATCH -e ./${SOLVER}_${ORDER}_${FACT}_${N}_${STEPS}.err        #stderr (%x=jobname, %j=jobid)
#SBATCH -D ./                 #Initial working directory
#SBATCH --partition=s.tok     #Queue/Partition
#SBATCH --qos=s.tok.standard  #Quality of Service (see below): s.tok.short, s.tok.standard, s.tok.long, tok.debug
#SBATCH --nodes=1             #Total number of nodes
#SBATCH --ntasks-per-node=1   #MPI tasks per node
#SBATCH --cpus-per-task=1     #CPUs per task for OpenMP
#SBATCH --mem-per-cpu=5G      #Set memory requirement
#SBATCH --time=00:15:00       #Wall clock limit

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
# For pinning threads correctly:
export OMP_PLACES=cores

# Run the program:
srun $BIN $N $SOLVER $TOL $STEPS $OUT $ORDER $FACT
EOF
else 
sbatch <<EOF
#!/bin/bash -l
#SBATCH -J $SOLVER            #Job name
#SBATCH -o ./${SOLVER}_${N}_${TOL}_${STEPS}.out        #stdout (%x=jobname, %j=jobid)
#SBATCH -e ./${SOLVER}_${N}_${TOL}_${STEPS}.err        #stderr (%x=jobname, %j=jobid)
#SBATCH -D ./                 #Initial working directory
#SBATCH --partition=s.tok     #Queue/Partition
#SBATCH --qos=s.tok.standard  #Quality of Service (see below): s.tok.short, s.tok.standard, s.tok.long, tok.debug
#SBATCH --nodes=1             #Total number of nodes
#SBATCH --ntasks-per-node=1   #MPI tasks per node
#SBATCH --cpus-per-task=1     #CPUs per task for OpenMP
#SBATCH --mem-per-cpu=5G   #Set memory requirement
#SBATCH --time=00:15:00       #Wall clock limit

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
# For pinning threads correctly:
export OMP_PLACES=cores

# Run the program:
srun $BIN $N $SOLVER $TOL $STEPS $OUT $ORDER $FACT
EOF
fi

