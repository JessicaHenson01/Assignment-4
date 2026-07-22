#!/bin/bash
python run.py \
  --mode train \
  --frame_dir HMDB51 \
  --n_classes 51 \
  --batch_size 4 \
  --model_type slow_fusion \
  --cnn_backbone resnet50 \
  --pretrained True \
  --fr_per_vid 16 \
  --rnn_n_layers 2 \
  --dropout 0.4 \
  --learning_rate 0.00003 \
  --n_epochs 30