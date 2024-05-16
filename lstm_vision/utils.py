import gc
import json
import logging
import os
import random
import subprocess
from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from torch import Tensor
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    IterableDataset,
    random_split,
)
from torchvision import datasets, transforms

from LSTM_model import LSTM


def total_norm__grads(model: nn.Module) -> float:
    """
    Calculate total norm of gradients.

    Args:
        model: Model for which we want to check whether gradient clipping is
            necessary.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # in case 2-norm is clipped
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    return total_norm


def cleanup():
    """
    Cleanup the distributed environment.
    """
    dist.destroy_process_group()


def setup(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        world_size: Number of processes participating in the job.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = "localhost"  # NOTE: might have to be adjusted
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def get_model(
    input_size: int,
    num_layers: int,
    hidden_size: int,
    num_classes: int,
    sequence_length: int,
    bidirectional: bool,
    dropout_rate: float,
    device: torch.device | int,
    use_ddp: bool = False,
) -> nn.Module:
    """

    Args:
        input_size: input is assumed to be in shape `(N, 1, H, W)`,
            where `W` is the input size
        num_layers: number of hidden layers for the NN
        hidden_size: number of features in hidden state `h`
        num_classes: number of classes our LSTM is supposed to predict,
            `10` for MNIST
        sequence_length: input is of shape `(N, sequence_length, input_size)`
        bidirectional: if `True`, use bidirectional LSTM
        dropout_rate: dropout rate for the dropout layer
        device: Device on which the code is executed. Can also be an int
            representing the GPU ID.
        use_ddp: Whether to use DDP.

    Returns:
        Model.
    """

    # define model
    model = LSTM(
        input_size=input_size,
        sequence_length=sequence_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_classes=num_classes,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
    )
    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[device])

    return model


def retrieve_args(parser: ArgumentParser) -> Namespace:
    """
    Retrieve and parse the args; some args might have been passed in a JSON
    config file.

    Returns:
        Argparse options.
    """
    args = parser.parse_args()

    if args.config is not None:
        if os.path.exists(args.config):
            assert args.config.endswith(".json"), (
                f"Config file should be a JSON file, but is a '{args.config}' "
                "file."
            )
            with open(args.config, "r") as f:
                config_args = json.load(f)  # type: dict

            # List all registered arguments
            registered_args = {action.dest for action in parser._actions}

            # Check if all config keys are known to the parser
            unknown_args = set(config_args) - registered_args

            # Check for unknown arguments
            if unknown_args:
                raise ValueError(
                    f"Unknown argument(s) in JSON config: {unknown_args}"
                )

            parser.set_defaults(**config_args)
            args = parser.parse_args()
        else:
            raise ValueError(f"Config file '{args.config}' not found.")

    check_args(args)

    return args


def check_args(args: Namespace) -> None:
    """
    Check provided arguments and print them to CLI.

    Args:
        args: Arguments provided by the user.
    """

    # create saving dir if non-existent
    os.makedirs(args.saving_path, exist_ok=True)

    if args.num_additional_cps > args.num_epochs:
        args.num_additional_cps = args.num_epochs

    assert args.compile_mode in [
        None,
        "default",
        "reduce-overhead",
        "max-autotune",
    ], (
        f"``{args.compile_mode}`` is not a valid compile mode in "
        "``torch.compile()``."
    )
    if args.pin_memory and not torch.cuda.is_available():
        args.pin_memory = False  # pinned memory only available for GPUs
    if args.pin_memory:
        assert args.num_workers > 0, (
            "With pinned memory, ``num_workers > 0`` should be chosen, cf. "
            "https://stackoverflow.com/questions/55563376/pytorch-how"
            "-does-pin-memory-work-in-dataloader"
        )
    assert 0 <= args.dropout_rate < 1, (
        "``dropout_rate`` should be chosen between 0 (inclusive) and 1 "
        f"(exclusive), but is {args.dropout_rate}."
    )
    assert 0 <= args.label_smoothing < 1, (
        "``label_smoothing`` should be chosen between 0 (inclusive) and 1 "
        f"(exclusive), but is {args.label_smoothing}."
    )
    assert 0 < args.train_split < 1, (
        "``train_split`` should be chosen between 0 and 1, "
        f"but is {args.train_split}."
    )


def get_datasets(
    channels_img: int, train_split: float
) -> Tuple[
    datasets.VisionDataset, datasets.VisionDataset, datasets.VisionDataset
]:
    """
    Get the train, val and test datasets.

    Args:
        channels_img: Number of channels in the input images.
        train_split: Percentage of the training set to use for training.

    Returns:
        Train, val and test datasets.
    """

    # define data transformation:
    trafo = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(channels_img)],
                std=[0.5 for _ in range(channels_img)],
            ),
        ]
    )

    full_train_dataset = datasets.MNIST(
        root="",
        train=True,
        transform=trafo,
        target_transform=None,
        download=True,
    )  # `60`k images

    num__train_samples = int(train_split * len(full_train_dataset))
    train_subset, val_subset = random_split(
        dataset=full_train_dataset,
        lengths=[
            num__train_samples,
            len(full_train_dataset) - num__train_samples,
        ],
    )
    test_dataset = datasets.MNIST(
        root="",
        train=False,
        transform=trafo,
        target_transform=None,
        download=True,
    )

    return train_subset, val_subset, test_dataset


def seed_worker(worker_id: int) -> None:
    """
    Seed the worker for the dataloader. Function copy-pasted from [1].

    Args:
        worker_id: Worker ID.

    References:
        [1] https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    train_dataset: datasets.VisionDataset,
    val_dataset: datasets.VisionDataset,
    test_dataset: datasets.VisionDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_ddp: bool = False,
    seed_number: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the dataloaders for the train, validation and test set.

    Args:
        train_dataset: Training set.
        val_dataset: Validation set.
        test_dataset: Test set.
        batch_size: Batch size.
        num_workers: Number of subprocesses used in the dataloaders.
        pin_memory: Whether tensors are copied into CUDA pinned memory.
        use_ddp: Whether to use DDP.
        seed_number: Seed number for the `torch.Generator` object if provided.

    Returns:
        Train, val and test loader
    """

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=DistributedSampler(train_dataset) if use_ddp else None,
        shuffle=False if use_ddp else True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed_number),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=DistributedSampler(val_dataset) if use_ddp else None,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed_number),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def get_git_info() -> None:
    """Get the git branch name and the git SHA."""
    try:
        c = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            timeout=10,
            stdout=subprocess.PIPE,
        )
        git_branch_name = c.stdout.decode().strip()
        c = subprocess.run(
            ["git", "rev-parse", "HEAD"], timeout=10, stdout=subprocess.PIPE
        )
        git_sha = c.stdout.decode().strip()
        logging.info(f"Git branch: {git_branch_name}, commit: {git_sha}\n")
    except subprocess.TimeoutExpired as e:
        logging.exception(e)
        logging.warn("Git info not found. Moving right along...")


def start_timer(device: torch.device | int) -> float:
    """
    Start the timer.

    Args:
        device: Device on which the code is executed. Can also be an int
            representing the GPU ID.

    Returns:
        Time at which the training started.
    """
    gc.collect()

    # check if device is a ``torch.device`` object; if not, assume it's an
    # int and convert it
    if not isinstance(device, torch.device):
        device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else "cpu"
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()

    return perf_counter()


def log_training_stats(
    start_time: float,
    energy_consump: float,
    device: torch.device | int,
    local_msg: str = "",
) -> float:
    """
    End the timer and print the time it took to execute the code as well as the
    maximum memory used by tensors.

    Args:
        start_time: Time at which the training started.
        energy_consump: Energy consumption of the entire training in Joules.
        device: Device on which the code was executed. Can also be an int
            representing the GPU ID.
        local_msg: Local message to print.

    Returns:
        Time it took to execute the code.
    """

    # check if device is a ``torch.device`` object; if not, assume it's an
    # int and convert it
    if not isinstance(device, torch.device):
        device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else "cpu"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_diff = perf_counter() - start_time

    msg = f"{local_msg}\n\tTotal execution time = {time_diff:.3f} [sec]"
    if device.type == "cuda":
        msg += (
            f"\n\tEnergy consumption of entire training = "
            f"{energy_consump / 1e3:.3f} [kJ]"
            f"\n\tMax memory used by tensors = "
            f"{torch.cuda.max_memory_allocated(device=device) / 1024**2:.3f} "
            "[MB]"
        )
    logging.info(msg)

    return time_diff


def format_line(
    mode: str,
    epoch: int,
    current_samples: int,
    total_samples: int,
    percentage: float,
    loss: Tensor,
) -> None:
    assert mode.lower() in ["train", "val"]

    # calculate maximum width for each part
    max_epoch_width = len(f"{mode.capitalize()} epoch: {epoch}")
    max_sample_info_width = len(f"[{total_samples} / {total_samples} (100 %)]")

    # format each part
    epoch_str = f"{mode.capitalize()} epoch: {epoch}".ljust(max_epoch_width)
    padded__current_sample = str(current_samples).zfill(
        len(str(total_samples))
    )
    sample_info_str = f"[{padded__current_sample} / {total_samples} ({percentage:06.2f} %)]".ljust(
        max_sample_info_width
    )
    loss_str = f"Loss: {loss:.4f}"

    return f"{epoch_str}  {sample_info_str}  {loss_str}"


def print__batch_info(
    mode: str,
    batch_idx: int,
    loader: DataLoader,
    epoch: int,
    loss: Tensor,
    frequency: Optional[int] = 1,
) -> None:
    """
    Print the current batch information.

    Params:
        mode: Mode in which the model is in. Either "train" or "val".
        batch_idx: Batch index.
        loader: Train or validation Dataloader.
        epoch: Current epoch.
        loss: Loss of the current batch.
        frequency: Frequency at which to print the batch info.
    """
    assert mode.lower() in ["train", "val"]

    if frequency is not None and batch_idx % frequency == 0:
        current_samples = (batch_idx + 1) * loader.batch_size
        if (
            not isinstance(loader.dataset, IterableDataset)
            and batch_idx == len(loader) - 1
        ):
            current_samples = len(loader.dataset)

        # https://stackoverflow.com/questions/5384570/how-can-i-count-the-number-of-items-in-an-arbitrary-iterable-such-as-a-generato
        total_samples = (
            sum(1 for _ in loader.dataset)
            if isinstance(loader.dataset, IterableDataset)
            else len(loader.dataset)
        )
        prog_perc = 100 * current_samples / total_samples

        formatted_line = format_line(
            mode=mode,
            epoch=epoch,
            current_samples=current_samples,
            total_samples=total_samples,
            percentage=prog_perc,
            loss=loss,
        )
        logging.info(f"{formatted_line}")


def load_checkpoint(
    model: nn.Module,
    checkpoint: dict[torch.Tensor, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Load an existing checkpoint of the model to continue training.

    Args:
        model: NN for which state dict is loaded.
        checkpoint: Checkpoint dictionary.
        optimizer: Optimizer for which state dict is loaded.
    """
    model.load_state_dict(state_dict=checkpoint["state_dict"])
    loading_msg = "=> Checkpoint loaded."

    if optimizer is not None:
        optimizer.load_state_dict(state_dict=checkpoint["optimizer"])

    if "epoch" in checkpoint.keys():
        loading_msg += f" It had been saved at epoch {checkpoint['epoch']}."
    elif "step" in checkpoint.keys():
        loading_msg += f" It had been saved at step {checkpoint['step']}."

    if "val_loss" in checkpoint.keys():
        loading_msg += f" Validation loss: {checkpoint['val_loss']:.4f}."

    if "val_acc" in checkpoint.keys():
        loading_msg += (
            f" Validation accuracy: {100 * checkpoint['val_acc']:.2f} %."
        )
    logging.info(loading_msg)


def save_checkpoint(state: Dict, filename: str = "my_checkpoint.pt") -> None:
    """
    Save checkpoint.

    Args:
        state: State of model and optimizer in a dictionary.
        filename: The name of the checkpoint.
    """
    log_msg = f"\n=> Saving checkpoint '{filename}' "
    if "val_loss" in state.keys():
        log_msg += (
            f"corresponding to a validation loss of {state['val_loss']:.4f} "
        )
    if "val_acc" in state.keys():
        log_msg += (
            f"and a validation accuracy of {100 * state['val_acc']:.2f} % "
        )
    if "epoch" in state.keys():
        log_msg += f"at epoch {state['epoch']}."
    logging.info(log_msg)

    torch.save(state, filename)


def count_parameters(model: nn.Module) -> None:
    """Print the number of parameters per module.

    Args:
        model: Model for which we want the total number of parameters.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logging.info(table)
