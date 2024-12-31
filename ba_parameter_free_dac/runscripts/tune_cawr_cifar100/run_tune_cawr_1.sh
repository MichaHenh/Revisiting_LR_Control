#!/bin/bash

#SBATCH --job-name=tune_cawr_cifar100_1
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=512G
#SBATCH --time=120:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output %x.out
#SBATCH --error %x.err

module load Miniforge3

conda activate PFDAC

cd ../..
python tune_cawr.py --config-name=tune_cawr_cifar100 seed=1