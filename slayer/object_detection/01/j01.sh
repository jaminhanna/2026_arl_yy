#!/bin/sh
#SBATCH --account=ISAAC-UTK0319
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=j01.out
#SBATCH --error=j01.err

module load anaconda3/2024.06
source $ANACONDA3_SH
conda activate lava-dl
python train_sdnn.py \
  -model tiny_yolov3_str \
  -load slayer \
  -num_workers 16 \
  -epoch 200 \
  -lr 0.0001 \
  -lrf 0.01 \
  -warmup 40 \
  -lambda_coord 2 \
  -lambda_noobj 4 \
  -lambda_obj 1.8 \
  -lambda_cls 1 \
  -lambda_iou 2.25 \
  -alpha_iou 0.8 \
  -clip 1 \
  -label_smoothing 0.03 \
  -tgt_iou_thr 0.5 \
  -aug_prob 0.4 \
  -track_iter 100 \
  -sparsity \
  -sp_lam 0.01 \
  -sp_rate 0.01 \
  -dataset FRED \
  -path data/fred \
  -b 2 \
  -verbose
conda deactivate
