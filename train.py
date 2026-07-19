"""Training and validation loops for video classifiers."""

import copy
import os

import torch
import wandb
from tqdm import tqdm


def train(
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    optim_model_wts_dir,
    n_epochs=30,
):
    """Train a model and retain the checkpoint with best validation accuracy."""
    loss_hist = {"train": [], "val": []}
    acc_hist = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(
        model.state_dict()
    )
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        learning_rates = get_learning_rates(
            optimizer
        )

        print(
            f"Epoch {epoch + 1}/{n_epochs}; "
            f"Current learning rates {learning_rates}"
        )

        model.train()

        # Keep frozen ResNet stages and their BatchNorm statistics fixed.
        model.base_model.eval()
        model.base_model.layer4.train()

        train_loss, train_accuracy = get_epoch_loss(
            model,
            criterion,
            dataloaders["train"],
            device,
            optimizer,
        )
        loss_hist["train"].append(train_loss)
        acc_hist["train"].append(train_accuracy)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(
                model,
                criterion,
                dataloaders["val"],
                device,
            )

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(
                model.state_dict()
            )
            best_model_path = os.path.join(
                optim_model_wts_dir,
                "best_model_wts.pt",
            )
            torch.save(
                best_model_wts,
                best_model_path,
            )
            print(
                "Best model weights are updated "
                f"at epoch {epoch + 1}!"
            )

        loss_hist["val"].append(val_loss)
        acc_hist["val"].append(val_accuracy)

        scheduler.step(val_loss)

        log_values = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        }

        for index, learning_rate in enumerate(
            get_learning_rates(optimizer)
        ):
            log_values[
                f"learning_rate/group_{index}"
            ] = learning_rate

        wandb.log(log_values)

        print(
            f"train loss: {train_loss:.6f}, "
            f"val loss: {val_loss:.6f}, "
            f"accuracy: {100 * val_accuracy:.2f}"
        )
        print("-" * 60)
        print()

    model.load_state_dict(best_model_wts)
    return model, loss_hist, acc_hist


def get_learning_rates(optimizer):
    """Return the learning rate for each optimizer parameter group."""
    return [
        parameter_group["lr"]
        for parameter_group in optimizer.param_groups
    ]


def batch_correct_preds(output, target):
    """Count correct top-1 predictions in a batch."""
    predictions = output.argmax(
        dim=1,
        keepdim=True,
    )
    return predictions.eq(
        target.view_as(predictions)
    ).sum().item()


def get_batch_loss(
    model,
    criterion,
    output,
    target,
    optimizer=None,
):
    """Compute batch loss and optionally update model parameters."""
    loss = criterion(output, target)

    with torch.no_grad():
        correct_predictions = batch_correct_preds(
            output,
            target,
        )

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

    return loss.item(), correct_predictions


def get_epoch_loss(
    model,
    criterion,
    dataloader,
    device,
    optimizer=None,
):
    """Compute dataset-average loss and accuracy."""
    running_loss = 0.0
    running_correct = 0
    processed_examples = 0

    for x_batch, y_batch in tqdm(dataloader):
        if x_batch is None:
            continue

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)
        batch_loss, batch_correct = get_batch_loss(
            model,
            criterion,
            output,
            y_batch,
            optimizer,
        )

        running_loss += batch_loss
        running_correct += batch_correct
        processed_examples += y_batch.size(0)

    if processed_examples == 0:
        raise RuntimeError(
            "No valid examples were processed."
        )

    loss = running_loss / float(
        processed_examples
    )
    accuracy = running_correct / float(
        processed_examples
    )

    return loss, accuracy
