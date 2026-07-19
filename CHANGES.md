# Motion Transformer Update

Replace the corresponding project files with:

- `models.py`
- `video_datasets.py`
- `utils.py`
- `train.py`
- `run.py`
- `train.sh`

## What changes

1. Adds `MotionTransformerClassifier`.
2. Builds an explicit motion stream from consecutive ResNet feature differences.
3. Replaces the LSTM temporal module with a two-layer Transformer encoder.
4. Adds learned positional embeddings and a classification token.
5. Randomly samples one frame from each temporal segment during training.
6. Uses deterministic uniform sampling during validation and test.
7. Applies one crop, flip, brightness change, and contrast change consistently
   to every frame in a training clip.
8. Keeps the existing `LRCN` class available as a baseline.
9. Logs every optimizer parameter-group learning rate.

## Important

Old LRCN checkpoints will not load into the motion Transformer. Train this model
from scratch using the included `train.sh`.

## Smoke test

```bash
python - <<'PY'
import torch
from models import MotionTransformerClassifier

model = MotionTransformerClassifier(
    n_classes=51,
    pretrained=False,
    cnn_model="resnet50",
    transformer_dim=256,
    transformer_heads=4,
    transformer_layers=2,
    transformer_ff_dim=1024,
    dropout_rate=0.25,
    max_frames=16,
)

sample = torch.randn(2, 16, 3, 224, 224)

model.eval()
with torch.no_grad():
    output = model(sample)

print(output.shape)
PY
```

Expected:

```text
torch.Size([2, 51])
```

## One-epoch trial

Temporarily change `--n_epochs 30` to `--n_epochs 1`, then run:

```bash
bash train.sh
```

Confirm that the first batch and validation pass finish before starting the full
experiment.
