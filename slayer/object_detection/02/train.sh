source .venv/bin/activate

python tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/train_sdnn.py \
         -verbose \
         -epoch 200 \
         -dataset FRED \
         -path /local_scratch/jhanna8/datasets/FRED \
         -output_dir 02

deactivate
