#!/bin/bash

python run.py \
  --mode train \
  --frame_dir HMDB51 \
  --n_classes 51 \
  --model_type motion_transformer \
  --cnn_backbone resnet50 \
  --pretrained True \
  --fr_per_vid 16 \
  --transformer_dim 256 \
  --transformer_heads 4 \
  --transformer_layers 2 \
  --transformer_ff_dim 1024 \
  --batch_size 2 \
  --dropout 0.25 \
  --learning_rate 0.0001 \
  --n_epochs 30
