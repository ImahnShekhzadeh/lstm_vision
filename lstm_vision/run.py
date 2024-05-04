import logging
import os
import sys
from argparse import Namespace
from datetime import datetime as dt

import torch
import wandb
from torch import multiprocessing as mp
from torch import optim
from torchinfo import summary

from evaluate import check_accuracy, get_confusion_matrix
from options import get_parser
from train import train_and_validate
from utils import (
    cleanup,
    count_parameters,
    get_dataloaders,
    get_datasets,
    get_git_info,
    get_model,
    load_checkpoint,
    retrieve_args,
    save_checkpoint,
    setup,
)


def main(
    rank: int | torch.device,
    world_size: int,
    args: Namespace,
) -> None:
    """
    Main function.

    Args:
        rank: rank of the current process
        world_size: number of processes
        args: command line arguments
    """

    # set random seed, each process gets different seed
    if args.seed_number is not None:
        torch.manual_seed(args.seed_number + rank)

    if args.use_ddp:
        setup(
            rank=rank,
            world_size=world_size,
        )

    # get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        channels_img=args.channels_img,
        train_split=args.train_split,
    )

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_ddp=args.use_ddp,
    )

    # define sequence length, input size of LSTM and number of classes based
    # on input data
    seq_length = test_loader.dataset[0][0].shape[1]
    inp_size = test_loader.dataset[0][0].shape[2]
    num_classes = len(test_loader.dataset.classes)

    # get model
    model = get_model(
        input_size=inp_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        sequence_length=seq_length,
        bidirectional=args.bidirectional,
        dropout_rate=args.dropout_rate,
        device=rank,
        use_ddp=args.use_ddp,
    )

    # get git info, setup Weights & Biases, print # data and model summary
    if rank in [0, torch.device("cpu")]:
        get_git_info()
        wandb_logging = args.wandb__api_key is not None

        logging.info(
            f"# Train:val:test samples: {len(train_loader.dataset)}"
            f":{len(val_loader.dataset)}:{len(test_loader.dataset)}\n"
        )
        logging.info(
            f"\n{summary(model, (args.batch_size, seq_length, inp_size))}\n"
        )
    else:
        wandb_logging = False

    # compile model if specified
    if args.compile_mode is not None:
        logging.info(f"\nCompiling model in ``{args.compile_mode}`` mode...\n")
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)

    # Optimizer:
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Set network to train mode:
    model.train()

    if args.loading_path is not None:
        if rank == torch.device("cpu"):
            map_location = {"cuda:0": "cpu"}
        else:
            map_location = {"cuda:0": f"cuda:{rank}"}

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint=torch.load(
                args.loading_path, map_location=map_location
            ),
        )

    # Train the network:
    checkpoint = train_and_validate(
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        num_grad_accum_steps=args.num_grad_accum_steps,
        rank=rank,
        use_amp=args.use_amp,
        train_loader=train_loader,
        val_loader=val_loader,
        timestamp=args.timestamp,
        num_additional_cps=args.num_additional_cps,
        saving_path=args.saving_path,
        label_smoothing=args.label_smoothing,
        freq_output__train=args.freq_output__train,
        freq_output__val=args.freq_output__val,
        max_norm=args.max_norm,
        wandb_logging=wandb_logging,
    )

    if rank in [0, torch.device("cpu")]:
        # save model and optimizer state dicts
        save_checkpoint(
            state=checkpoint,
            filename=os.path.join(
                args.saving_path,
                f"lstm_cp_{args.timestamp}.pt",
            ),
        )

    # destroy process group if DDP was used (for clean exit)
    if args.use_ddp:
        cleanup()

    if rank in [0, torch.device("cpu")]:
        count_parameters(model)  # TODO: rename, misleadig name

        # load checkpoint with lowest validation loss for final evaluation;
        # device does not need to be specified, since the checkpoint will be
        # loaded on the CPU or GPU with ID 0 depending on where the checkpoint
        # was saved
        load_checkpoint(model=model, checkpoint=checkpoint)

        # check accuracy on train and test set and produce confusion matrix
        check_accuracy(
            train_loader,
            model,
            use_amp=args.use_amp,
            mode="train",
            device=rank,
        )
        check_accuracy(
            test_loader, model, use_amp=args.use_amp, mode="test", device=rank
        )
        get_confusion_matrix(
            num_classes,
            test_loader,
            model,
            use_amp=args.use_amp,
            saving_path=args.saving_path,
            device=rank,
            timestamp=args.timestamp,
        )

        if wandb_logging:
            wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = retrieve_args(parser)

    # Get timestamp
    args.timestamp = dt.now().strftime("%dp%mp%Y_%Hp%Mp%S")

    # Setup Weights & Biases
    if args.wandb__api_key is not None:
        wandb.login(key=args.wandb__api_key)
        wandb.init(
            project="lstm_vision",
            name=args.timestamp,
            config=args,
        )

    # Setup basic configuration for logging
    log_level = logging.INFO
    logging.basicConfig(
        filename=os.path.join(args.saving_path, f"run_{args.timestamp}.log"),
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create `StreamHandler` for stdout and add it to root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    logging.getLogger().addHandler(console_handler)

    if args.config is not None and os.path.exists(args.config):
        logging.info(f"Config file '{args.config}' found and loaded.")
    logging.info(args)

    # define world size (number of GPUs)
    world_size = torch.cuda.device_count()

    if torch.cuda.is_available():
        list_gpus = [torch.cuda.get_device_name(i) for i in range(world_size)]
        logging.info(f"\nGPU(s): {list_gpus}\n")

    if args.use_ddp and world_size > 1:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / world_size)
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        args.use_ddp = False
        main(
            rank=0 if world_size >= 1 else torch.device("cpu"),
            world_size=1,
            args=args,
        )
