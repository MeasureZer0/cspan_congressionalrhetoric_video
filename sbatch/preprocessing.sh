#!/bin/bash -l
#SBATCH -N 1               # Number of nodes. ALWAYS set to 1
#SBATCH -n 1               # Number of tasks. ALWAYS set to 1
#SBATCH -c 32              # Number of CPU cores. Can go as high as 128. Each additional CPU core adds around 1.9GB of RAM so to get more memory, add more CPU cores.
#SBATCH -t 1:0:0           # Number of hours to run (H:M:S). Change as needed.
#SBATCH -A cis220051-gpu   # The TDM account to charge for this. Don't change.
#SBATCH -p gpu             # Partition to use.
#SBATCH --gpus-per-node=1  # Must use just one GPU.  Do not change!

# These three lines "load" the TDM python.  Almost always keep them.
module use /anvil/projects/tdm/opt/core
module load tdm
module load python/seminar r/seminar

SCRIPT_DIR="$(dirname -- "$(readlink -f "${BASH_SOURCE}")")"
cd $SCRIPT_DIR/..

python3 preprocessing/crop_faces.py >logs/crop_faces.log 2>logs/crop_faces.err

# sbatch commands:
# sbatch preprocessing.sh
# squeue --me
# scancel <jobid>
