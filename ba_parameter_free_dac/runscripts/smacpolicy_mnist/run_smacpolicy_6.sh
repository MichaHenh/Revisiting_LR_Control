#!/bin/bash

#SBATCH --job-name=SMAC_policy_6
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output %x.out
#SBATCH --error %x.err

module load Miniconda3

conda activate PFDAC

cd ../..
python smac_policy.py --config-name=smacpolicy seed=6 -m