#!/bin/bash

#SBATCH --job-name=tune_cawr_cifar10_1
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output %x.out
#SBATCH --error %x.err

module load Miniforge3

conda activate PFDAC

cd ../..
python tune_cawr.py --config-name=tune_cawr_cifar10 seed=1 -m