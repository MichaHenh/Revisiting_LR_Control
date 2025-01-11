#!/bin/bash

#SBATCH --job-name=tune_cawr_libsvm_1
#SBATCH --partition=ai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=finn.micha.henheik@stud.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output %x.out
#SBATCH --error %x.err

module load Miniforge3

conda activate PFDAC

cd ../..
python tune_cawr.py --config-name=tune_cawr_libsvm dacbench_sgd_config.dataset_name=iris,sensorless,aloi,dna,letter,pendigits,vehicle,vowel,wine seed=1 -m