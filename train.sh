#!/bin/bash
python run.py \
  --mode train \
  --frame_dir HMDB51 \
  --n_classes 51 \
  --batch_size 4 \
  --model_type lrcn \
  --cnn_backbone resnet50 \
  --pretrained True \
  --fr_per_vid 16 \
  --rnn_hidden_size 256 \
  --rnn_n_layers 2 \
  --dropout 0.3 \
  --learning_rate 0.00003 \
  --n_epochs 30