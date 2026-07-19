"""Video utilities, clip-level transforms, and DataLoader construction."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from video_datasets import (
    collate_fn_r3d_18,
    collate_fn_rnn,
)


# pylint: disable=no-member
def get_frames(vid, n_frames=1):
    """Uniformly sample frames from a video file."""
    frames = []
    video_capture = cv2.VideoCapture(vid)

    if not video_capture.isOpened():
        print("Failed to open video:", vid)
        return frames, 0

    video_length = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if video_length <= 0:
        print("No frames found in video:", vid)
        video_capture.release()
        return frames, 0

    frame_indices = np.linspace(
        0,
        video_length - 1,
        n_frames,
        dtype=int,
    )
    frame_indices = set(frame_indices.tolist())

    for index in range(video_length):
        success, frame = video_capture.read()

        if not success:
            continue

        if index in frame_indices:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
            frames.append(frame)

    video_capture.release()
    return frames, video_length


def store_frames(frames, store_path):
    """Save RGB frames as JPEG images."""
    os.makedirs(store_path, exist_ok=True)

    for index, frame in enumerate(frames):
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR,
        )
        frame_path = os.path.join(
            store_path,
            f"frame{index}.jpg",
        )
        cv2.imwrite(frame_path, frame)


def transform_stats(model="lrcn"):
    """Return image dimensions and normalization values."""
    if model in {"lrcn", "motion_transformer"}:
        height, width = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == "3dcnn":
        height, width = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError(
            f"Undefined model type: {model}"
        )

    return height, width, mean, std


class ClipTransform:
    """Apply one spatial/color augmentation consistently to an entire clip."""

    def __init__(
        self,
        height,
        width,
        mean,
        std,
        training,
    ):
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.training = training

    def __call__(self, frames):
        if not frames:
            raise ValueError(
                "ClipTransform received an empty frame list."
            )

        if self.training:
            enlarged_height = self.height + 32
            enlarged_width = self.width + 32

            resized_frames = [
                transform_functional.resize(
                    frame,
                    [enlarged_height, enlarged_width],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                for frame in frames
            ]

            top, left, crop_height, crop_width = (
                transforms.RandomResizedCrop.get_params(
                    resized_frames[0],
                    scale=(0.80, 1.0),
                    ratio=(0.90, 1.10),
                )
            )

            apply_flip = random.random() < 0.5
            brightness_factor = random.uniform(
                0.85,
                1.15,
            )
            contrast_factor = random.uniform(
                0.85,
                1.15,
            )

            processed_frames = []

            for frame in resized_frames:
                frame = transform_functional.resized_crop(
                    frame,
                    top,
                    left,
                    crop_height,
                    crop_width,
                    [self.height, self.width],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )

                if apply_flip:
                    frame = transform_functional.hflip(
                        frame
                    )

                frame = (
                    transform_functional.adjust_brightness(
                        frame,
                        brightness_factor,
                    )
                )
                frame = (
                    transform_functional.adjust_contrast(
                        frame,
                        contrast_factor,
                    )
                )

                processed_frames.append(frame)
        else:
            processed_frames = [
                transform_functional.resize(
                    frame,
                    [self.height, self.width],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                for frame in frames
            ]

        tensors = []

        for frame in processed_frames:
            frame_tensor = transform_functional.to_tensor(
                frame
            )
            frame_tensor = transform_functional.normalize(
                frame_tensor,
                self.mean,
                self.std,
            )
            tensors.append(frame_tensor)

        return torch.stack(tensors)


def compose_data_transforms(
    height,
    width,
    mean,
    std,
):
    """Return clip-consistent training and evaluation transforms."""
    train_transform = ClipTransform(
        height=height,
        width=width,
        mean=mean,
        std=std,
        training=True,
    )
    validation_test_transform = ClipTransform(
        height=height,
        width=width,
        mean=mean,
        std=std,
        training=False,
    )

    return (
        train_transform,
        validation_test_transform,
    )


def _uses_sequence_layout(model):
    return model in {
        "lrcn",
        "motion_transformer",
    }


def train_val_dloaders(
    train_dataset,
    val_dataset,
    batch_size,
    model="lrcn",
):
    """Create training and validation DataLoaders."""
    collate_function = (
        collate_fn_rnn
        if _uses_sequence_layout(model)
        else collate_fn_r3d_18
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function,
    )
    validation_loader = DataLoader(
        val_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        collate_fn=collate_function,
    )

    return {
        "train": train_loader,
        "val": validation_loader,
    }


def test_dloaders(
    test_dataset,
    batch_size,
    model="lrcn",
):
    """Create a test DataLoader."""
    collate_function = (
        collate_fn_rnn
        if _uses_sequence_layout(model)
        else collate_fn_r3d_18
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2 * batch_size,
        shuffle=False,
        collate_fn=collate_function,
    )

    return {"test": test_loader}
