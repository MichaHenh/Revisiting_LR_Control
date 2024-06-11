#!/bin/bash

#SBATCH --job-name=smac_policy_benchmark
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output smac_policy_benchmark.out
#SBATCH --error smac_policy_benchmark.err

module load Miniconda3

conda activate BAPFDAC

python adam_smac_exp.py
