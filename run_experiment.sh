#!/bin/bash
### Bash script to start a slurm job on the Edinburgh cluster ###

source ~/.bashrc
set -e
SCRATCH_DISK=/disk/scratch
USER=s1686853
FILE_GENERATOR_NAME=$1
EXPERIMENT_FILE_NAME=$2
SAVE_NAME=$3
LOG_NAME=$4
EXP_NAME=$5
SAVE_PATH=/home/${USER}/${SAVE_NAME}
LOG_PATH=${SCRATCH_DISK}/${USER}/${LOG_NAME}

echo "Generating Experiment File"
mkdir -p ${SAVE_PATH}
python ${FILE_GENERATOR_NAME} ${EXPERIMENT_FILE_NAME} ${LOG_PATH} ${SAVE_PATH} ${EXP_NAME}

echo "Running parallel batch"
N_EXPERIMENTS=`cat ${EXPERIMENT_FILE_NAME} | wc -l`
MAX_PARALLEL_JOBS=60
sbatch --array=1-${N_EXPERIMENTS}%${MAX_PARALLEL_JOBS} setup_single_experiment.sh $EXPERIMENT_FILE_NAME