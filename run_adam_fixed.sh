#!/bin/bash

#SBATCH --job-name=cocob_benchmark
#SBATCH --partition=ai
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --array=10-12,18
#SBATCH --output test_array-job_%A_%a.out
#SBATCH --error test_array-job_%A_%a.err

module load Miniconda3

conda activate BAPFDAC

python adam_fixed_exp.py
