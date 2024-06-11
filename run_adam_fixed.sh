#!/bin/bash

#SBATCH --job-name=adam_fixed_benchmark
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output adam_fixed_benchmark.out
#SBATCH --error adam_fixed_benchmark.err

module load Miniconda3

conda activate BAPFDAC

python ba_parameter_free_dac/adam_fixed_exp.py
