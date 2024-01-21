import os
import sys
from datetime import datetime as dt
from math import ceil
from typing import Optional

import torch
from functions import (
    check_accuracy,
    cleanup,
    count_parameters,
    end_timer_and_print,
    get_dataloaders,
    get_model,
    load_checkpoint,
    produce_acc_plot,
    produce_and_print_confusion_matrix,
    produce_loss_plot,
    retrieve_args,
    save_checkpoint,
    setup,
    train_and_validate,
)
from torch import optim
from torchinfo import summary
from train_options import get_parser


def main(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> None:
    """
    Main function.

    Args:
        rank: rank of the current process
        world_size: number of processes
    """
    parser = get_parser()
    args = retrieve_args(parser)

    if args.seed_number is not None:
        torch.manual_seed(args.seed_number)

    if args.use_ddp:
        setup(
            rank=rank,
            world_size=world_size,
        )

    # Set device, get gpu, check number of available gpus
    # and whether DDP is supposed to be used:
    num_gpus = torch.cuda.device_count()
    use_gpu = num_gpus >= 1
    device = torch.device("cuda:0" if use_gpu else "cpu")
    gpu_id = None

    if use_gpu:
        gpu_id = torch.cuda.current_device()
        list_gpus = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"\nGPU(s): {list_gpus}\n")

    if args.use_ddp:
        if num_gpus > 1:
            print("Using DistributedDataParallel.")
        else:
            args.use_ddp = False
            print(
                "DistributedDataParallel not used, since two or more GPUs are "
                "NOT available."
            )

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        channels_img=args.channels_img,
        train_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_ddp=args.use_ddp,
    )

    # define sequence length and input size of LSTM based on input data
    seq_length = test_loader.dataset[0][0].shape[1]
    inp_size = test_loader.dataset[0][0].shape[2]

    # get model and print summary
    model = get_model(
        input_size=inp_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_classes=len(test_loader.dataset.classes),
        sequence_length=seq_length,
        bidirectional=args.bidirectional,
        dropout_rate=args.dropout_rate,
        use_ddp=args.use_ddp,
        gpu_id=gpu_id,
    )
    summary(model, (args.batch_size, seq_length, inp_size))

    # compile model if specified
    if args.compile_mode is not None:
        print(f"\nCompiling model in ``{args.compile_mode}`` mode...\n")
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)

    # Optimizer:
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    # Set network to train mode:
    model.train()

    if args.loading_path is not None:
        load_checkpoint(
            model=model, optimizer=optimizer, checkpoint=args.loading_path
        )

    # Train the network:
    (
        start_time,
        checkpoint,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
    ) = train_and_validate(
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        use_amp=args.use_amp,
        train_loader=train_loader,
        val_loader=val_loader,
        freq_output__train=args.freq_output__train,
        freq_output__val=args.freq_output__val,
        max_norm=args.max_norm,
    )

    epoch_str = "epoch"
    if args.num_epochs > 1:
        epoch_str += "s"
    num_iters = (
        ceil(len(train_loader.dataset) / args.batch_size) * args.num_epochs
    )
    end_timer_and_print(
        start_time=start_time,
        device=device,
        local_msg=(
            f"Training {args.num_epochs} {epoch_str} ({num_iters} iterations)"
        ),
    )
    save_checkpoint(
        state=checkpoint,
        filename=os.path.join(
            args.saving_path,
            f"lstm_cp_{dt.now().strftime('%dp%mp%Y_%Hp%M')}.pt",
        ),
    )

    # destroy process group if DDP was used (for clean exit)
    if args.use_ddp:
        cleanup()

    count_parameters(model)  # TODO: rename, misleadig name
    produce_loss_plot(
        args.num_epochs, train_losses, val_losses, args.saving_path
    )
    produce_acc_plot(args.num_epochs, train_accs, val_accs, args.saving_path)

    # check accuracy on train and test set and produce confusion matrix
    load_checkpoint(model=model, checkpoint=checkpoint)
    check_accuracy(train_loader, model, mode="train", device=device)
    check_accuracy(test_loader, model, mode="test", device=device)
    produce_and_print_confusion_matrix(
        len(test_loader.dataset.classes),  # TODO: write `num_clases` instead
        test_loader,
        model,
        args.saving_path,
        device,
    )


if __name__ == "__main__":
    main()
