#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4 #number of cores per node (max 64)
#SBATCH --gres=gpu:3
#SBATCH --time=00:15:00 # hours:minutes:seconds
#SBATCH --partition=aa100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=put_your_email_address_here
#SBATCH --output=modtfr1_%j.out #output file (%j adds job id)

echo "== Starting Job =="

echo "== Changing directory =="
cd put_path_to_directory_here #e.g. /projects/<username>/mydir

echo "== Loading conda module =="
module purge
module load anaconda

echo "== Activating conda environment =="
conda activate r-tf2150py3118

echo "== Starting R =="
Rscript --vanilla transfer_learning_hpc.R

echo "== End of Job =="