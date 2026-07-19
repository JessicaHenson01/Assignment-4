"""
Module: video_datasets.py

Utilities for loading, splitting, sampling, and batching video-frame datasets.
"""

import glob
import os
import re

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional
from tqdm import tqdm


def get_frame_number(frame_path):
    """Return the last numeric value in a frame filename."""
    filename = os.path.basename(frame_path)
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else -1


class VideoDataset(Dataset):
    """
    Dataset for videos stored as directories of extracted frames.

    Training clips use random temporal-segment sampling.
    Validation clips use deterministic uniform sampling.
    Test clips can optionally return beginning, middle, and ending views.
    """

    def __init__(
        self,
        vid_dataset,
        fr_per_vid,
        transforms=None,
        training=False,
        multi_clip_eval=False,
    ):
        self.dataset = list(vid_dataset)
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.training = training
        self.multi_clip_eval = multi_clip_eval

    def __len__(self):
        """Return the number of videos."""
        return len(self.dataset)

    def _sample_frame_indices(self, frame_count):
        """Select exactly ``self.fpv`` frame indices."""
        if frame_count <= 0:
            raise ValueError(
                "frame_count must be greater than zero."
            )

        if frame_count < self.fpv:
            return np.rint(
                np.linspace(
                    0,
                    frame_count - 1,
                    num=self.fpv,
                )
            ).astype(int)

        if not self.training:
            return np.rint(
                np.linspace(
                    0,
                    frame_count - 1,
                    num=self.fpv,
                )
            ).astype(int)

        boundaries = np.linspace(
            0,
            frame_count,
            num=self.fpv + 1,
            dtype=int,
        )
        selected_indices = []

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            if end <= start:
                frame_index = min(
                    start,
                    frame_count - 1,
                )
            else:
                frame_index = np.random.randint(
                    start,
                    end,
                )

            selected_indices.append(frame_index)

        return np.asarray(
            selected_indices,
            dtype=int,
        )

    def _get_three_view_indices(self, frame_count):
        """Return beginning, middle, and ending clip indices."""
        if frame_count <= 0:
            raise ValueError(
                "frame_count must be greater than zero."
            )

        if frame_count <= self.fpv:
            repeated_indices = np.rint(
                np.linspace(
                    0,
                    frame_count - 1,
                    num=self.fpv,
                )
            ).astype(int)

            return [
                repeated_indices.copy(),
                repeated_indices.copy(),
                repeated_indices.copy(),
            ]

        maximum_start = frame_count - self.fpv
        start_positions = [
            0,
            maximum_start // 2,
            maximum_start,
        ]

        return [
            np.arange(
                start,
                start + self.fpv,
                dtype=int,
            )
            for start in start_positions
        ]

    def _load_clip(self, frame_paths, selected_indices):
        """Load and transform one clip into shape ``(T, C, H, W)``."""
        frames = []

        for frame_index in selected_indices:
            frame_path = frame_paths[int(frame_index)]

            with Image.open(frame_path) as frame:
                frames.append(
                    frame.convert("RGB").copy()
                )

        if self.transforms is not None:
            return self.transforms(frames)

        return torch.stack(
            [
                transform_functional.to_tensor(frame)
                for frame in frames
            ]
        )

    def __getitem__(self, idx):
        """
        Return one sample and its class label.

        Single-clip mode returns ``(T, C, H, W)``.
        Multi-clip evaluation returns ``(3, T, C, H, W)``.
        """
        video_path, frame_label = self.dataset[idx]

        frame_paths = sorted(
            glob.glob(
                os.path.join(
                    video_path,
                    "*.jpg",
                )
            ),
            key=get_frame_number,
        )

        if not frame_paths:
            raise FileNotFoundError(
                f"No JPEG frames found in: {video_path}"
            )

        if self.multi_clip_eval:
            if self.training:
                raise ValueError(
                    "multi_clip_eval cannot be used with training=True."
                )

            view_indices = self._get_three_view_indices(
                len(frame_paths)
            )
            clip_tensors = [
                self._load_clip(
                    frame_paths,
                    selected_indices,
                )
                for selected_indices in view_indices
            ]

            return (
                torch.stack(clip_tensors),
                int(frame_label),
            )

        selected_indices = self._sample_frame_indices(
            len(frame_paths)
        )
        frames_tensor = self._load_clip(
            frame_paths,
            selected_indices,
        )

        return frames_tensor, int(frame_label)


def load_dataset(frame_dir):
    """Load video directories containing at least one JPEG frame."""
    class_names = [
        class_name
        for class_name in sorted(os.listdir(frame_dir))
        if os.path.isdir(
            os.path.join(frame_dir, class_name)
        )
    ]

    label_dict = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    video_dataset = {}
    skipped_videos = []

    print("Loading video dataset....")

    for class_name in tqdm(class_names):
        class_path = os.path.join(
            frame_dir,
            class_name,
        )

        for video_name in sorted(os.listdir(class_path)):
            video_path = os.path.join(
                class_path,
                video_name,
            )

            if not os.path.isdir(video_path):
                continue

            frame_paths = glob.glob(
                os.path.join(video_path, "*.jpg")
            )

            if not frame_paths:
                skipped_videos.append(video_path)
                continue

            video_dataset[video_path] = label_dict[class_name]

    print(
        f"Loaded {len(video_dataset)} videos; "
        f"skipped {len(skipped_videos)} empty directories."
    )

    if skipped_videos:
        print("Skipped empty video directories:")

        for video_path in skipped_videos:
            print(f"  {video_path}")

    return video_dataset, label_dict


def dataset_split(
    vid_dataset,
    tr_ratio,
    ts_ratio,
    seed=0,
):
    """Split the dataset using stratified sampling."""
    vid_paths = np.array(
        list(vid_dataset.keys())
    )
    vid_labels = np.array(
        list(vid_dataset.values())
    )

    print(
        "Splitting train/validation/test datasets...."
    )

    test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=ts_ratio,
        random_state=seed,
    )
    train_val_indices, test_indices = next(
        test_splitter.split(
            vid_paths,
            vid_labels,
        )
    )

    test_paths = vid_paths[test_indices]
    test_labels = vid_labels[test_indices]
    train_val_paths = vid_paths[train_val_indices]
    train_val_labels = vid_labels[train_val_indices]

    test_dataset = list(
        zip(
            test_paths.tolist(),
            test_labels.tolist(),
        )
    )

    validation_ratio = 1 - tr_ratio - ts_ratio
    validation_weight = (
        validation_ratio
        / (tr_ratio + validation_ratio)
    )

    validation_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_weight,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        validation_splitter.split(
            train_val_paths,
            train_val_labels,
        )
    )

    train_dataset = list(
        zip(
            train_val_paths[train_indices].tolist(),
            train_val_labels[train_indices].tolist(),
        )
    )
    validation_dataset = list(
        zip(
            train_val_paths[validation_indices].tolist(),
            train_val_labels[validation_indices].tolist(),
        )
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def collate_fn_r3d_18(batch):
    """Stack fixed-length clips for a 3D CNN."""
    valid_samples = [
        (images, label)
        for images, label in batch
        if images.numel() > 0
    ]

    if not valid_samples:
        return None, None

    images_batch, label_batch = zip(*valid_samples)
    images_tensor = torch.stack(images_batch)
    images_tensor = torch.transpose(
        images_tensor,
        2,
        1,
    )
    labels_tensor = torch.tensor(
        label_batch,
        dtype=torch.long,
    )

    return images_tensor, labels_tensor


def collate_fn_rnn(batch):
    """Stack fixed-length clips for sequence models."""
    valid_samples = [
        (images, label)
        for images, label in batch
        if images.numel() > 0
    ]

    if not valid_samples:
        return None, None

    images_batch, label_batch = zip(*valid_samples)
    images_tensor = torch.stack(images_batch)
    labels_tensor = torch.tensor(
        label_batch,
        dtype=torch.long,
    )

    return images_tensor, labels_tensor