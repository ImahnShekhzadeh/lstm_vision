import logging
import os
from copy import deepcopy
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from utils import (
    end_timer_and_print,
    print__batch_info,
    save_checkpoint,
    start_timer,
)


def train_and_validate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    num_grad_accum_steps: int,
    rank: int | torch.device,
    use_amp: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    timestamp: str,
    num_additional_cps: int = 0,
    saving_path: str = None,
    label_smoothing: float = 0.0,
    freq_output__train: Optional[int] = 10,
    freq_output__val: Optional[int] = 10,
    max_norm: Optional[float] = None,
    wandb_logging: bool = False,
) -> Dict:
    """
    Train and validate the model.

    Args:
        model: Model to train.
        optimizer: Optimizer to use.
        num_epochs: Number of epochs to train the model.
        num_grad_accum_steps: Number of gradient accumulation steps.
        rank: Device on which the code is executed.
        use_amp: Whether to use automatic mixed precision.
        train_loader: Dataloader for the training set.
        val_loader: Dataloader for the validation set.
        timestamp: Timestamp of the current run.
        num_additional_cps: Number of checkpoints to save (one is always saved
            at the lowest validation loss)
        saving_path: Directory path to save the checkpoints.
        label_smoothing: Amount of smoothing to be applied when calculating
            loss.
        freq_output__train: Frequency at which to print the training info.
        freq_output__val: Frequency at which to print the validation info.
        max_norm: Maximum norm of the gradients.
        wandb_logging: API key for Weights & Biases.

    Returns:
        Checkpoint of the model corresponding to the lowest validation loss.
    """

    # check saving path
    if num_additional_cps >= 1:
        assert (
            saving_path is not None
        ), "Please provide a valid saving path for the additional checkpoints"
        os.makedirs(saving_path, exist_ok=True)

    # define loss function
    cce_mean = nn.CrossEntropyLoss(
        reduction="mean", label_smoothing=label_smoothing
    )

    start_time = start_timer(device=rank)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    min_val_loss = float("inf")

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        t0 = start_timer(device=rank)

        trainingLoss_perEpoch, num_correct, num_samples = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=cce_mean,
            num_grad_accum_steps=num_grad_accum_steps,
            scaler=scaler,
            rank=rank,
            use_amp=use_amp,
            epoch=epoch,
            max_norm=max_norm,
            freq_output__train=freq_output__train,
        )
        train_losses.append(
            np.sum(trainingLoss_perEpoch, axis=0) / len(train_loader.dataset)
        )
        train_accs.append(num_correct / num_samples)

        valLoss_perEpoch, num_correct, num_samples = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=cce_mean,
            rank=rank,
            use_amp=use_amp,
            epoch=epoch,
            freq_output__val=freq_output__val,
        )
        val_losses.append(
            np.sum(valLoss_perEpoch, axis=0) / len(val_loader.dataset)
        )
        val_accs.append(num_correct / num_samples)

        # update checkpoint dict if val loss has decreased
        if val_losses[epoch] < min_val_loss:
            min_val_loss = val_losses[epoch]
            checkpoint_best = {
                "state_dict": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
                "val_loss": val_losses[epoch],
                "val_acc": val_accs[epoch],
                "epoch": epoch,
            }

        # save additional checkpoints
        if (
            (num_additional_cps >= 1)
            and ((epoch + 1) % (num_epochs // num_additional_cps) == 0)
            and rank in [0, torch.device("cpu")]
        ):
            checkpoint = {
                "state_dict": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
                "val_loss": val_losses[epoch],
                "val_acc": val_accs[epoch],
                "epoch": epoch,
            }
            save_checkpoint(
                state=checkpoint,
                filename=os.path.join(
                    saving_path,
                    f"cp_epoch_{epoch}_{timestamp}.pt",
                ),
            )

        if rank in [0, torch.device("cpu")]:
            # log to Weights & Biases
            if wandb_logging:
                wandb.log(
                    {
                        "train_loss": train_losses[epoch],
                        "val_loss": val_losses[epoch],
                        "train_acc": train_accs[epoch],
                        "val_acc": val_accs[epoch],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

            logging.info(
                f"\nEpoch {epoch}: {perf_counter() - t0:.3f} [sec]\t"
                f"Mean train/val loss: {train_losses[epoch]:.4f}/"
                f"{val_losses[epoch]:.4f}\tTrain/val acc: "
                f"{1e2 * train_accs[epoch]:.2f} %/{1e2 * val_accs[epoch]:.2f} "
                "%\n"
            )

    # number of iterations per device
    num_iters = len(train_loader) * num_epochs

    if rank in [0, torch.device("cpu")]:
        end_timer_and_print(
            start_time=start_time,
            device=rank,
            local_msg=(
                f"Training {num_epochs} epochs ({num_iters} iterations)"
            ),
        )

    return checkpoint_best


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._WeightedLoss,
    num_grad_accum_steps: int,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    rank: int | torch.device,
    use_amp: bool,
    epoch: int,
    max_norm: Optional[float] = None,
    freq_output__train: Optional[int] = 10,
) -> Tuple[List[float], int, int]:
    """
    Train model for one epoch.

    Args:
        train_loader: Dataloader for the training set.
        model: Model to train.
        optimizer: Optimizer to use.
        loss_fn: Loss function.
        num_grad_accum_steps: Number of gradient accumulation steps.
        scaler: GradScaler for automatic mixed precision.
        rank: Device on which the code is executed.
        use_amp: Whether to use automatic mixed precision.
        epoch: Current epoch index.
        max_norm: Maximum norm of the gradients.
        freq_output__train: Frequency at which to print the training info.

    Returns:
        Training loss for all batches for the single epoch, number of correct
        predictions, and number of samples.
    """

    trainingLoss_perEpoch = []
    num_correct, num_samples = 0, 0  # for calculating accuracy
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        labels = labels.to(rank)

        with autocast(
            device_type=labels.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(images.squeeze_(dim=1).to(rank))  # `(N, 10)`
            loss = loss_fn(output, labels) / num_grad_accum_steps

        scaler.scale(loss).backward()
        if ((batch_idx + 1) % num_grad_accum_steps == 0) or (
            batch_idx + 1 == len(train_loader)
        ):
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            if max_norm is not None:
                scaler.unscale_(optimizer)
                for param_group in optimizer.param_groups:
                    clip_grad_norm_(param_group["params"], max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        trainingLoss_perEpoch.append(
            loss.cpu().item() * num_grad_accum_steps * output.shape[0]
        )

        # calculate accuracy
        with torch.no_grad():
            batch_size = output.shape[0]
            max_indices = output.argmax(dim=1, keepdim=False)
            num_correct += (max_indices == labels).sum().cpu().item()
            num_samples += batch_size

        if rank in [0, torch.device("cpu")]:
            print__batch_info(
                batch_idx=batch_idx,
                loader=train_loader,
                epoch=epoch,
                loss=loss * num_grad_accum_steps,
                mode="train",
                frequency=freq_output__train,
            )

    return trainingLoss_perEpoch, num_correct, num_samples


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.modules.loss._WeightedLoss,
    rank: int | torch.device,
    use_amp: bool,
    epoch: int,
    freq_output__val: Optional[int] = 10,
) -> Tuple[List[float], int, int]:
    """
    Validate model for one epoch.

    Args:
        model: Model to validate.
        val_loader: Dataloader for the validation set.
        loss_fn: Loss function.
        rank: Device on which the code is executed.
        use_amp: Whether to use automatic mixed precision.
        epoch: Current epoch index.
        freq_output__val: Frequency at which to print the validation info.

    Returns:
        Validation loss for all batches for the single epoch, number of correct
        predictions, and number of samples.
    """
    with torch.no_grad():
        valLoss_perEpoch = []
        val_num_correct, val_num_samples = 0, 0
        model.eval()

        for val_batch_idx, (val_images, val_labels) in enumerate(val_loader):
            val_labels = val_labels.to(rank)

            with autocast(
                device_type=val_labels.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                val_output = model(
                    val_images.squeeze_(dim=1).to(rank)
                )  # `[N, C]`
                val_loss = loss_fn(val_output, val_labels)

            valLoss_perEpoch.append(val_loss.item() * val_output.shape[0])

            # calculate accuracy
            val_max_indices = val_output.argmax(dim=1, keepdim=False)
            val_num_correct += (
                (val_max_indices == val_labels).cpu().sum().item()
            )
            batch_size = val_output.shape[0]
            val_num_samples += batch_size

            if rank in [0, torch.device("cpu")]:
                print__batch_info(
                    batch_idx=val_batch_idx,
                    loader=val_loader,
                    epoch=epoch,
                    loss=val_loss,
                    mode="val",
                    frequency=freq_output__val,
                )

    return valLoss_perEpoch, val_num_correct, val_num_samples
