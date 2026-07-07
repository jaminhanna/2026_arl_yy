#!/bin/sh
#SBATCH --account=ISAAC-UTK0319
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=out/j04b.out
#SBATCH --error=err/j04b.err

module load anaconda3/2024.06
source $ANACONDA3_SH
module load gcc
conda activate tennlab_slayer
python scripts/j04b.py
conda deactivate
