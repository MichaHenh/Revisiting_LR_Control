#!/bin/bash

#SBATCH --job-name=SMAC_policy_cifar100_4
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=512G
#SBATCH --time=56:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output %x.out
#SBATCH --error %x.err

module load Miniconda3

conda activate PFDAC

cd ../..
python smac_policy.py --config-name=create_smacpolicy_cifar100 seed=4 -m