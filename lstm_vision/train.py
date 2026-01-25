import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import autocast, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from typeguard import typechecked
from zeus.monitor import ZeusMonitor

from utils import (
    get__gradient_scaler,
    get__save_or_log,
    log__after_epoch,
    print__batch_info,
    save_checkpoint,
    save_cp_log__after_training,
    start_timer,
)

str__cuda_0 = "cuda:0"


@dataclass
class TrainingConfig:
    """Configuration for the training and validation process."""

    # Training loop params
    num_epochs: int  # number of epochs to train model
    num_grad_accum_steps: int  # number of gradient accumulation steps
    world_size: int  # number of processes participating in distributed training
    use_amp: bool  # whether automatic mixed precision is used
    max_norm: Optional[float] = None  # maximum norm of gradients
    label_smoothing: float = (
        0.0  # amount of smoothing applied when calculating loss
    )

    # Checkpoint and saving params
    num_additional_cps: int = 0
    # number of additional checkpoints (besides the one for lowest val loss)
    saving_path: Optional[str] = None  # directory path for checkpoints
    saving_name_best_cp: Optional[str] = None  # saving name of best checkpoint

    # Logging and monitoring params
    freq_output__train: Optional[
        int
    ] = 10  # freq at which training info is printed
    freq_output__val: Optional[int] = 10  # freq at which val info is printed
    wandb_logging: bool = False  # whether logging to Weights & Biases occurs


@typechecked
def train_and_validate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int | torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    train_sampler: Optional[Sampler] = None,
) -> None:
    """
    Train and validate the model. The code will always save a checkpoint
    corresponding to the model on rank 0 with lowest validation loss.

    Args:
        model: Model to train.
        optimizer: Optimizer to use.
        rank: Device on which the code is executed.
        train_loader: Dataloader for the training set.
        val_loader: Dataloader for the validation set.
        cfg: Training configuration containing all training, checkpoint/saving and
            logging parameters.
        train_sampler: Sampler for the training set.
    """

    if cfg.num_additional_cps >= 1:
        assert (
            cfg.saving_path is not None
        ), "Please provide a valid saving path for the additional checkpoints"
        os.makedirs(cfg.saving_path, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss(
        reduction="sum", label_smoothing=cfg.label_smoothing
    )
    scaler = get__gradient_scaler(rank=rank, use_amp=cfg.use_amp)
    min_val_loss = float("inf")
    save_or_log = get__save_or_log(rank=rank)

    # measure energy consumption (rank 0 already measures energy consumption
    # of all GPUs)
    if save_or_log and rank != torch.device("cpu"):
        monitor = ZeusMonitor(gpu_indices=list(range(cfg.world_size)))
        monitor.begin_window("training")

    start_time = start_timer(device=rank)
    for epoch in range(cfg.num_epochs):
        start_time__epoch = start_timer(device=rank)

        if train_sampler is not None:
            # necessary to ensure shuffling of the data
            # https://pytorch.org/docs/stable/data.html
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_grad_accum_steps=cfg.num_grad_accum_steps,
            scaler=scaler,
            rank=rank,
            epoch=epoch,
            max_norm=cfg.max_norm,
            freq_output__train=cfg.freq_output__train,
        )
        # mean loss per sample over all GPUs; we could alternatively use
        # `torch.distributed.reduce()` to sum the losses over all GPUs
        train_loss *= cfg.world_size / len(train_loader.dataset)

        val_loss, val_acc = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            rank=rank,
            use_amp=cfg.use_amp,
            epoch=epoch,
            freq_output__val=cfg.freq_output__val,
        )
        val_loss *= cfg.world_size / len(val_loader.dataset)

        if val_loss < min_val_loss and save_or_log:
            min_val_loss = val_loss
            checkpoint_best = {
                "state_dict": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
            }

        if (
            (cfg.num_additional_cps >= 1)
            and ((epoch + 1) % (cfg.num_epochs // cfg.num_additional_cps) == 0)
            and save_or_log
        ):
            checkpoint = {
                "state_dict": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
            }
            saving_name = f"cp_epoch_{epoch}.pt"

            save_checkpoint(
                state=checkpoint,
                filename=os.path.join(
                    cfg.saving_path,
                    saving_name,
                ),
            )

        if save_or_log:
            log__after_epoch(
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                epoch=epoch,
                start_time__epoch=start_time__epoch,
                wandb_logging=cfg.wandb_logging,
            )

    if save_or_log:
        save_cp_log__after_training(
            rank=rank,
            monitor=monitor,
            train_loader=train_loader,
            num_epochs=cfg.num_epochs,
            world_size=cfg.world_size,
            start_time=start_time,
            checkpoint_best=checkpoint_best,
            saving_path=cfg.saving_path,
            saving_name_best_cp=cfg.saving_name_best_cp,
        )


@typechecked
def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._WeightedLoss,
    num_grad_accum_steps: int,
    scaler: torch.amp.GradScaler,
    rank: int | torch.device,
    epoch: int,
    max_norm: Optional[float] = None,
    freq_output__train: Optional[int] = 10,
) -> Tuple[float, float]:
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
        epoch: Current epoch index.
        max_norm: Maximum norm of the gradients.
        freq_output__train: Frequency at which to print the training info.

    Returns:
        Summsed training loss for all batches for the single epoch, train
        accuracy on rank 0 or CPU.
    """

    epoch_loss, num_correct, num_samples = 0, 0, 0
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        labels = labels.to(rank)

        with autocast(
            device_type=labels.device.type,
            dtype=torch.float16,
            enabled=scaler._enabled,
        ):
            output = model(images.squeeze_(dim=1).to(rank))  # `(N, 10)`
            loss = loss_fn(output, labels) / num_grad_accum_steps

        batch_size = output.shape[0]
        scaler.scale(loss / batch_size).backward()
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

        epoch_loss += loss.item() * num_grad_accum_steps

        with torch.no_grad():
            max_indices = output.argmax(dim=1, keepdim=False)
            num_correct += (max_indices == labels).sum().cpu().item()
            num_samples += batch_size

        if rank in [
            0,
            torch.device(str__cuda_0),
            torch.device("cuda"),
            torch.device("cpu"),
        ]:
            print__batch_info(
                batch_idx=batch_idx,
                loader=train_loader,
                epoch=epoch,
                loss=loss * num_grad_accum_steps / batch_size,
                mode="train",
                frequency=freq_output__train,
            )

    return epoch_loss, num_correct / num_samples


@torch.no_grad()
@typechecked
def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.modules.loss._WeightedLoss,
    rank: int | torch.device,
    use_amp: bool,
    epoch: int,
    freq_output__val: Optional[int] = 10,
) -> Tuple[float, float]:
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
        Summsed training loss for all batches for the single epoch, train
        accuracy on rank 0 or CPU.
    """
    epoch_loss, val_num_correct, val_num_samples = 0, 0, 0
    model.eval()

    for val_batch_idx, (val_images, val_labels) in enumerate(val_loader):
        val_labels = val_labels.to(rank)

        with autocast(
            device_type=val_labels.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            val_output = model(val_images.squeeze_(dim=1).to(rank))  # `[N, C]`
            val_loss = loss_fn(val_output, val_labels)

        epoch_loss += val_loss.item()
        val_max_indices = val_output.argmax(dim=1, keepdim=False)
        val_num_correct += (val_max_indices == val_labels).sum().item()
        batch_size = val_output.shape[0]
        val_num_samples += batch_size

        if rank in [
            0,
            torch.device(str__cuda_0),
            torch.device("cuda"),
            torch.device("cpu"),
        ]:
            print__batch_info(
                batch_idx=val_batch_idx,
                loader=val_loader,
                epoch=epoch,
                loss=val_loss / batch_size,
                mode="val",
                frequency=freq_output__val,
            )

    return epoch_loss, val_num_correct / val_num_samples
