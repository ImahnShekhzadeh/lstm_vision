import os
import sys
from datetime import datetime as dt
from math import ceil

import torch
from functions import (
    check_accuracy,
    check_args,
    count_parameters,
    end_timer_and_print,
    get_dataloaders,
    load_checkpoint,
    produce_acc_plot,
    produce_and_print_confusion_matrix,
    produce_loss_plot,
    save_checkpoint,
    train_and_validate,
)
from LSTM_model import LSTM
from torch import optim
from torchinfo import summary
from train_options import TrainOptions


def main() -> None:
    """Main function."""
    args = TrainOptions().args

    check_args(args)

    if args.seed_number is not None:
        torch.manual_seed(args.seed_number)

    # Set device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        channels_img=args.channels_img,
        train_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # define sequence length and input size of LSTM based on input data
    seq_length = test_loader.dataset[0][0].shape[1]
    inp_size = test_loader.dataset[0][0].shape[2]

    # define model
    model = LSTM(
        input_size=inp_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_classes=len(test_loader.dataset.classes),
        sequence_length=seq_length,
        bidirectional=args.bidirectional,
        dropout_rate=args.dropout_rate,
    ).to(device)

    # print model summary
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
        len(test_loader.dataset.classes),
        test_loader,
        model,
        args.saving_path,
        device,
    )


if __name__ == "__main__":
    main()
