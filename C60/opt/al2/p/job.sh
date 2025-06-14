#!/bin/bash
#SBATCH -J adl_p
#SBATCH -p cpu
#SBATCH -w b003
#SBATCH -c 18
#SBATCH -G 0
#SBATCH -o Job%J_%x_%N_out
#SBATCH -e Job%J_%x_%N_err

hostname
source ~/.bashrc
python opt.py > opt.out
# python convert.py > convert.out
