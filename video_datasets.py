"""Dataset loading and clip-level frame sampling."""

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
    """Load a fixed-length clip from each extracted-frame directory."""

    def __init__(
        self,
        vid_dataset,
        fr_per_vid,
        transforms=None,
        training=False,
    ):
        self.dataset = list(vid_dataset)
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.training = training

    def __len__(self):
        return len(self.dataset)

    def _sample_indices(self, frame_count):
        """Choose exactly self.fpv indices."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")

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

        # Random segment sampling: choose one frame from each temporal segment.
        boundaries = np.linspace(
            0,
            frame_count,
            num=self.fpv + 1,
            dtype=int,
        )
        indices = []

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            if end <= start:
                index = min(start, frame_count - 1)
            else:
                index = np.random.randint(start, end)

            indices.append(index)

        return np.asarray(indices, dtype=int)

    def __getitem__(self, idx):
        video_path, frame_label = self.dataset[idx]

        frame_paths = sorted(
            glob.glob(
                os.path.join(video_path, "*.jpg")
            ),
            key=get_frame_number,
        )

        if not frame_paths:
            raise FileNotFoundError(
                f"No JPEG frames found in: {video_path}"
            )

        selected_indices = self._sample_indices(
            len(frame_paths)
        )
        selected_paths = [
            frame_paths[index]
            for index in selected_indices
        ]

        frames = []
        for frame_path in selected_paths:
            with Image.open(frame_path) as image:
                frames.append(
                    image.convert("RGB").copy()
                )

        if self.transforms is not None:
            frames_tensor = self.transforms(frames)
        else:
            frames_tensor = torch.stack(
                [
                    transform_functional.to_tensor(frame)
                    for frame in frames
                ]
            )

        return frames_tensor, int(frame_label)


def load_dataset(frame_dir):
    """Load nonempty video directories and assign integer class labels."""
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

            video_dataset[video_path] = (
                label_dict[class_name]
            )

    print(
        f"Loaded {len(video_dataset)} videos; "
        f"skipped {len(skipped_videos)} empty directories."
    )

    return video_dataset, label_dict


def dataset_split(
    vid_dataset,
    tr_ratio,
    ts_ratio,
    seed=0,
):
    """Create stratified train, validation, and test splits."""
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

    train_paths = {path for path, _ in train_dataset}
    validation_paths = {
        path for path, _ in validation_dataset
    }
    test_paths = {path for path, _ in test_dataset}

    assert train_paths.isdisjoint(validation_paths)
    assert train_paths.isdisjoint(test_paths)
    assert validation_paths.isdisjoint(test_paths)

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
    images_tensor = images_tensor.transpose(1, 2)
    labels_tensor = torch.tensor(
        label_batch,
        dtype=torch.long,
    )

    return images_tensor, labels_tensor


def collate_fn_rnn(batch):
    """Stack fixed-length clips as (batch, time, channels, height, width)."""
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
