"""Train or evaluate a video action-recognition model."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import (
    LRCN,
    MotionTransformerClassifier,
)
from test import (
    get_confusion_matrix,
    get_test_report,
    test,
)
from train import train
from utils import (
    compose_data_transforms,
    test_dloaders,
    train_val_dloaders,
    transform_stats,
)
from video_datasets import (
    VideoDataset,
    dataset_split,
    load_dataset,
)


def str_to_bool(value):
    """Parse a command-line Boolean safely."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected a Boolean value."
    )


def args_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Classification Training"
    )

    parser.add_argument(
        "-fd",
        "--frame_dir",
        required=True,
        help="Directory containing extracted video frames",
    )
    parser.add_argument(
        "-trs",
        "--train_size",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "-tss",
        "--test_size",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-fpv",
        "--fr_per_vid",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-nc",
        "--n_classes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        help="Checkpoint path for evaluation",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        choices=[
            "lrcn",
            "motion_transformer",
            "3dcnn",
        ],
        default="motion_transformer",
    )
    parser.add_argument(
        "-cnn",
        "--cnn_backbone",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ],
        default="resnet50",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        type=str_to_bool,
        default=True,
    )
    parser.add_argument(
        "-rhs",
        "--rnn_hidden_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "-rnl",
        "--rnn_n_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--transformer_dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--transformer_heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transformer_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--transformer_ff_dim",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["train", "eval"],
        default="train",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for newly initialized temporal layers",
    )
    parser.add_argument(
        "-ne",
        "--n_epochs",
        type=int,
        default=30,
    )

    return parser.parse_args()


def create_model(args):
    """Create the selected network."""
    if args.model_type == "lrcn":
        return LRCN(
            hidden_size=args.rnn_hidden_size,
            n_layers=args.rnn_n_layers,
            dropout_rate=args.dropout,
            n_classes=args.n_classes,
            pretrained=args.pretrained,
            cnn_model=args.cnn_backbone,
        )

    if args.model_type == "motion_transformer":
        return MotionTransformerClassifier(
            n_classes=args.n_classes,
            pretrained=args.pretrained,
            cnn_model=args.cnn_backbone,
            transformer_dim=args.transformer_dim,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_ff_dim=args.transformer_ff_dim,
            dropout_rate=args.dropout,
            max_frames=args.fr_per_vid,
        )

    raise ValueError(
        "The current run.py does not construct a 3D CNN."
    )


def configure_trainable_backbone(model):
    """Fine-tune only the final ResNet stage."""
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    for parameter in model.base_model.layer4.parameters():
        parameter.requires_grad = True


def create_optimizer(model, args):
    """Create differential-learning-rate parameter groups."""
    backbone_parameters = list(
        model.base_model.layer4.parameters()
    )

    temporal_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(
            "base_model.layer4"
        )
    ]

    return optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.learning_rate * 0.1,
            },
            {
                "params": temporal_parameters,
                "lr": args.learning_rate,
            },
        ],
        weight_decay=1e-4,
    )


def main(args):
    """Train or evaluate the selected model."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    height, width, mean, std = transform_stats(
        args.model_type
    )
    train_transform, eval_transform = (
        compose_data_transforms(
            height,
            width,
            mean,
            std,
        )
    )

    model = create_model(args)
    configure_trainable_backbone(model)

    if args.mode == "train":
        video_dataset, _ = load_dataset(
            args.frame_dir
        )
        train_split, val_split, test_split = (
            dataset_split(
                video_dataset,
                args.train_size,
                args.test_size,
            )
        )

        splits = {
            "train": np.array(
                train_split,
                dtype=object,
            ),
            "val": np.array(
                val_split,
                dtype=object,
            ),
            "test": np.array(
                test_split,
                dtype=object,
            ),
        }
        np.save(
            "./splits.npy",
            splits,
            allow_pickle=True,
        )

        train_dataset = VideoDataset(
            train_split,
            args.fr_per_vid,
            train_transform,
            training=True,
        )
        val_dataset = VideoDataset(
            val_split,
            args.fr_per_vid,
            eval_transform,
            training=False,
        )

        dataloaders = train_val_dloaders(
            train_dataset,
            val_dataset,
            args.batch_size,
            args.model_type,
        )

        loss_function = nn.CrossEntropyLoss(
            reduction="sum",
            label_smoothing=0.1,
        )
        optimizer = create_optimizer(
            model,
            args,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        os.makedirs(
            "./models",
            exist_ok=True,
        )

        wandb.init(
            project=(
                "assignment-4-video-action-recognition"
            ),
            config=vars(args),
        )

        model.to(device)
        train(
            dataloaders,
            model,
            loss_function,
            optimizer,
            scheduler,
            device,
            "./models",
            args.n_epochs,
        )
        wandb.finish()
        return

    splits = np.load(
        "./splits.npy",
        allow_pickle=True,
    ).item()

    test_split = [
        (sample[0], int(sample[1]))
        for sample in splits["test"]
    ]
    test_dataset = VideoDataset(
        test_split,
        args.fr_per_vid,
        eval_transform,
        training=False,
    )
    dataloaders = test_dloaders(
        test_dataset,
        args.batch_size,
        args.model_type,
    )

    if not args.ckpt:
        raise ValueError(
            "--ckpt is required in eval mode."
        )

    model.load_state_dict(
        torch.load(
            args.ckpt,
            map_location=device,
        )
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    targets, outputs, test_loss, test_accuracy = test(
        model,
        dataloaders["test"],
        device,
        criterion,
    )

    wandb.init(
        project=(
            "assignment-4-video-action-recognition"
        ),
        job_type="evaluation",
        config=vars(args),
    )
    wandb.log(
        {
            "test/loss": test_loss,
            "test/accuracy": test_accuracy,
        }
    )
    wandb.finish()

    print(f"Test loss: {test_loss:.6f}")
    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # Optional reports:
    # print(get_test_report(targets, outputs, all_cats))
    # print(
    #     get_confusion_matrix(
    #         targets,
    #         outputs,
    #         labels_dict,
    #         all_cats,
    #     )
    # )


if __name__ == "__main__":
    main(args_parser())
