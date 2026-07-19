"""
Module: utils.py

This module provides helper functions for video processing and data transformations 
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn

import random
import torch

from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

# pylint: disable=no-member

def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Total number of frames in the video.
            
    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        print("No frames found in video:", vid)
        v_cap.release()
        return frames, 0
    frame_idx = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    for idx in range(v_len):
        success, frame = v_cap.read()
        if not success:
            continue
        if idx in frame_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    for idx, frame in enumerate(frames):
        print("processing")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, "frame{}.jpg".format(idx))
        cv2.imwrite(path_to_frame, frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std

class ClipTransform:
    """
    Apply the same spatial and color augmentation to every frame.

    This preserves temporal consistency within a video clip.
    """

    def __init__(
        self,
        height,
        width,
        mean,
        std,
        training=False,
    ):
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.training = training

    def __call__(self, frames):
        """
        Transform a list of PIL images into a tensor.

        Returns:
            Tensor shaped (T, C, H, W).
        """
        if not frames:
            raise ValueError(
                "ClipTransform received an empty frame list."
            )

        if self.training:
            processed_frames = (
                self._apply_training_transform(frames)
            )
        else:
            processed_frames = (
                self._apply_evaluation_transform(frames)
            )

        frame_tensors = []

        for frame in processed_frames:
            frame_tensor = (
                transform_functional.to_tensor(frame)
            )
            frame_tensor = (
                transform_functional.normalize(
                    frame_tensor,
                    self.mean,
                    self.std,
                )
            )
            frame_tensors.append(frame_tensor)

        return torch.stack(frame_tensors)

    def _apply_training_transform(self, frames):
        """Apply one random augmentation configuration to a clip."""
        resize_height = self.height + 32
        resize_width = self.width + 32

        resized_frames = [
            transform_functional.resize(
                frame,
                [resize_height, resize_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            for frame in frames
        ]

        # Select one crop location for the entire clip.
        top, left, _, _ = (
            transforms.RandomCrop.get_params(
                resized_frames[0],
                output_size=(
                    self.height,
                    self.width,
                ),
            )
        )

        # Select augmentation values once per video.
        apply_flip = random.random() < 0.5

        brightness_factor = random.uniform(
            0.85,
            1.15,
        )
        contrast_factor = random.uniform(
            0.85,
            1.15,
        )
        saturation_factor = random.uniform(
            0.90,
            1.10,
        )

        processed_frames = []

        for frame in resized_frames:
            frame = transform_functional.crop(
                frame,
                top,
                left,
                self.height,
                self.width,
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
            frame = (
                transform_functional.adjust_saturation(
                    frame,
                    saturation_factor,
                )
            )

            processed_frames.append(frame)

        return processed_frames

    def _apply_evaluation_transform(self, frames):
        """Apply deterministic resizing to a validation/test clip."""
        return [
            transform_functional.resize(
                frame,
                [self.height, self.width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            for frame in frames
        ]

def compose_data_transforms(
    height,
    width,
    mean,
    std,
):
    """
    Create clip-consistent training and evaluation transforms.
    """
    train_transforms = ClipTransform(
        height=height,
        width=width,
        mean=mean,
        std=std,
        training=True,
    )

    val_test_transforms = ClipTransform(
        height=height,
        width=width,
        mean=mean,
        std=std,
        training=False,
    )

    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn'):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    if model == "lrcn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_rnn)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn)
    else:
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_r3d_18)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_r3d_18)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn'):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    if model == "lrcn":
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_rnn)
    else:
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_r3d_18)
    dataloaders = {'test': test_dl}
    return dataloaders

