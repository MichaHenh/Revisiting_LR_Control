#!/bin/bash

#SBATCH --job-name=cocob_benchmark_mnist
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output %x.out
#SBATCH --error %x.err

module load Miniconda3

conda activate PFDAC

python ../cli.py --config-name=cocob_mnist