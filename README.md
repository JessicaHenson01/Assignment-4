# Assignment 4: Video Action Recognition with HMDB51

This repository contains the completed code for Assignment 4 in Deep Learning with PyTorch. The project fixes an video-classification codebase and trains a Long-term Recurrent Convolutional Network (LRCN) on the HMDB51 human-action dataset.

The final submitted model uses a pretrained ResNet-50 backbone, a two-layer bidirectional LSTM, temporal attention, and multi-clip evaluation.

## Final Results

| Metric | Result |
|---|---:|
| Best validation accuracy | 72.54% |
| Test accuracy | Approximately 72% |
| Test loss | Approximately 1.60 |
| Number of classes | 51 |
| Frames per clip | 16 |

The final evaluation averages predictions from beginning, middle, and ending temporal views of each test video.

## Errors Found and Fixed

### 1. LSTM Dimension Bug

The inherited model passed tensors shaped `(batch, time, features)` into an LSTM that used PyTorch's default `batch_first=False` behavior. As a result, the batch dimension could be interpreted as the temporal dimension.

The fix:

- Enabled `batch_first=True`.
- Reshaped frame features explicitly to `(batch, time, features)`.
- Processed all frames through the CNN in one vectorized operation.
- Selected temporal outputs using the correct batch-first indexing.

### 2. Test-Set Leakage

The training workflow created train, validation, and test datasets and DataLoaders together. This made the held-out test set available inside the training workflow.

The fix:

- Training now creates only train and validation DataLoaders.
- The test split is saved to `splits.npy`.
- Test data is loaded only when running explicit evaluation.
- Assertions verify that train, validation, and test video paths are disjoint.

## Model-Level Improvements

### 1. Stronger ResNet Backbone

The configuration used a smaller ResNet backbone. The final model uses an ImageNet-pretrained ResNet-50 to extract stronger spatial features from each frame.

The earlier ResNet layers are frozen while later layers are fine-tuned using smaller learning rates.

### 2. Bidirectional LSTM with Temporal Attention

The temporal model was expanded to:

- Two LSTM layers.
- Bidirectional sequence processing.
- Hidden size of 256.
- Temporal attention over all frame-level LSTM outputs.
- Dropout regularization before classification.

Temporal attention allows the model to assign more weight to informative moments instead of using only the final LSTM output.

## Additional Improvements

- Clip-consistent augmentation applies the same crop, flip, and color transformation to every frame in a clip.
- Random temporal-segment sampling is used during training.
- Deterministic uniform sampling is used for validation and ordinary evaluation.
- Beginning, middle, and ending clip predictions are averaged during final evaluation.
- Frame files are numerically sorted.
- Short clips repeat real frames rather than introducing black padding.
- Empty frame directories are skipped.
- Training, validation, and test metrics are logged to Weights & Biases.
- The best checkpoint is selected using validation accuracy.
- Gradient clipping and learning-rate scheduling are used during training.

## Dataset

The project uses the HMDB51 dataset, which contains 51 categories.

The code expects pre-extracted frames organized as:

```text
HMDB51/
├── brush_hair/
│   ├── video_1/
│   │   ├── frame0.jpg
│   │   ├── frame1.jpg
│   │   └── ...
│   └── ...
├── cartwheel/
├── catch/
└── ...
```

Each action directory contains one directory per video, and each video directory contains extracted JPEG frames.

The dataset itself is not included in this repository.

## Environment Setup

A Conda environment is what I used.

```bash
conda create -n assignment4 python=3.12
conda activate assignment4
python -m pip install -r requirements.txt
```

On systems that encounter binary compatibility errors between NumPy, SciPy, and scikit-learn, install NumPy 1.26:

```bash
python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.17.1" \
  "scikit-learn==1.9.0"
```

Confirm the environment before running:

```bash
which python

python -c "import torch, numpy, scipy, sklearn; \
print('Torch:', torch.__version__); \
print('NumPy:', numpy.__version__); \
print('SciPy:', scipy.__version__); \
print('scikit-learn:', sklearn.__version__)"
```

The code supports CUDA, Apple Metal Performance Shaders (MPS), and CPU execution. Training on CPU will be considerably slower.

## Final Model Configuration

```text
Model type:             LRCN
CNN backbone:           ResNet-50
CNN pretraining:        ImageNet
Frames per video:       16
LSTM hidden size:       256
LSTM layers:            2
Bidirectional LSTM:     Yes
Temporal attention:     Yes
Dropout:                0.3
Number of classes:      51
Training batch size:    4
Evaluation batch size:  1
Epochs:                 30
```

## Training

Activate the project environment and make sure the extracted `HMDB51` directory is in the repository root or update `--frame_dir` in `train.sh`.

```bash
conda activate assignment4
bash train.sh
```

Training performs the following steps:

1. Loads all nonempty video-frame directories.
2. Creates stratified train, validation, and test splits.
3. Saves the fixed splits to `splits.npy`.
4. Verifies that all three splits are disjoint.
5. Trains using the train split.
6. Selects the best checkpoint using validation accuracy.
7. Logs training and validation metrics to Weights & Biases.

The saved checkpoint is written under the `models/` directory.

## Evaluation

The final evaluation script expects:

```text
models/best_resnet50_lrcn_clip_aug.pt
```

Run:

```bash
conda activate assignment4
bash eval.sh
```

Evaluation:

- Loads the previously saved test split.
- Loads the ResNet-50 LRCN checkpoint.
- Samples beginning, middle, and ending views.
- Averages the three model-logit tensors.
- Reports test loss and overall accuracy.
- Logs test metrics to Weights & Biases.

Model checkpoints are excluded from Git because of their file size. Place the trained checkpoint in `models/` before running evaluation.

## Weights & Biases

Training, validation, and test metrics are logged to the project:

[Assignment 4 Video Action Recognition](https://wandb.ai/jhenso13-johns-hopkins-university/assignment-4-video-action-recognition)

The best 30-epoch ResNet-50 LRCN training run was logged as `noble-fog-9`.

## Code Quality

Run Pylint across the Python source files with:

```bash
python -m pylint \
  models.py \
  run.py \
  run_training.py \
  test.py \
  train.py \
  utils.py \
  video_datasets.py
```

Generated files, local datasets, model checkpoints, Python caches, and local Weights & Biases files should not be committed.

## Project Structure

```text
.
├── README.md
├── eval.sh
├── models.py
├── requirements.txt
├── run.py
├── run_training.py
├── test.py
├── train.py
├── train.sh
├── utils.py
└── video_datasets.py
```

### Main Files

- `models.py`: ResNet-LSTM-attention model definition.
- `video_datasets.py`: Dataset loading, temporal sampling, and split utilities.
- `utils.py`: Transform, DataLoader, frame-extraction, and metric utilities.
- `train.py`: Training and validation loop.
- `test.py`: Single-clip and three-view evaluation.
- `run.py`: Main training and evaluation entry point.
- `train.sh`: Final training configuration.
- `eval.sh`: Final evaluation configuration.

## Reproducing the Final Experiment

```bash
conda activate assignment4
bash train.sh
bash eval.sh
```