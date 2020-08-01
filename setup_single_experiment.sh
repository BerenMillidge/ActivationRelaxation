#!/bin/bash

### BASH SCRIPT TO RUN SET OF EXPERIMENT GENERATED FROM EXPERIMENT FILE ON EDINBURGH CLUSTER ###

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:2

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=16000

#amount of tasks run in parallel on each node - this is just a test!
# #SBATCH --ntasks-per-node=1

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
#  #SBATCH --time=7-00:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
# #SBATCH --partition=PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]

source ~/.bashrc
set -e
echo "Setting up log files"
USER=s1686853
SCRATCH_DISK=/disk/scratch
log_path=${SCRATCH_DISK}/${USER}/activation_relaxation
mkdir -p ${log_path}

echo "Initializing Conda Environment"
CONDA_NAME=env
conda activate ${CONDA_NAME}
#mkdir -p ${log_path}


echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS
repo_home=/home/${USER}/ActivationRelaxation
mnist_path=${repo_home}/mnist_data
#src_path=${repo_home}/experiments/examples/mnist/data/input

# input data directory path on the scratch disk of the node
mnist_dest_path=${SCRATCH_DISK}/${USER}/mnist_data
mkdir -p ${mnist_dest_path}  # make it if required


# rsync data across from headnode to compute node
rsync --archive --update --compress --progress ${mnist_path}/ ${mnist_dest_path}
echo "Rsynced mnist"

echo "Running experiment command"
experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"