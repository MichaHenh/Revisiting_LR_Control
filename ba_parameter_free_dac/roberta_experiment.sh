#!/bin/bash

#SBATCH --job-name=RoBERTa_bookwiki_test
#SBATCH --partition=ai

#SBATCH --output %x.out
#SBATCH --error %x.err

#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=8  # Number of GPUs per node
#SBATCH --cpus-per-task=4  # Number of CPU cores per GPU
#SBATCH --gres=gpu:8  # Request 8 GPUs
#SBATCH --time=24:00:00  # Maximum runtime
#SBATCH --mem=128G  # Memory per node
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL

module load Miniconda3

conda activate PFDAC

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)  # Master node address
export MASTER_PORT=12345           # A free port for communication
export WORLD_SIZE=$SLURM_NTASKS    # Total number of processes
export RANK=$SLURM_PROCID          # Global rank of the process
export LOCAL_RANK=$SLURM_LOCALID   # Local rank of the process

python roberta_experiment.py