import gc
import os
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from prettytable import PrettyTable
from torch import Tensor, nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Timing utilities
start_time = None


def start_timer(device: torch.device) -> None:
    """Start the timer."""
    global start_time

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()

    start_time = perf_counter()


def end_timer_and_print(device: torch.device, local_msg: str) -> None:
    """
    End the timer and print the time it took to execute the code.

    Args:
        device (torch.device): Device on which the code was executed.
        local_msg (str): Local message to print.

    Returns:
        None
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = perf_counter()
    msg = (
        f"{local_msg}\n\tTotal execution time = {end_time - start_time:.3f} "
        f"[sec]"
    )
    if device.type == "cuda":
        msg += (
            f"\n\tMax memory used by tensors = "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.3f} [MB]"
        )
    print(msg)


def print__batch_info(
    batch_idx: int,
    loader: DataLoader,
    epoch: int,
    t_0: float,
    loss: Tensor,
    mode: str,
    frequency: int = 1,
):
    assert mode.lower() in ["train", "val"]
    assert type(frequency) == int

    if batch_idx % frequency == 0:
        print_statement = f"{mode.capitalize()} epoch: {epoch} "

        if batch_idx == len(loader) - 1:
            print_statement += (
                f"[{len(loader.dataset)} / {len(loader.dataset)} (100 %)] "
            )
        else:
            prog_perc = (
                100 * (batch_idx + 1) * loader.batch_size / len(loader.dataset)
            )
            print_statement += (
                f"[{(batch_idx + 1) * loader.batch_size} / "
                f"{len(loader.dataset)} ({prog_perc:.2f} %)] "
            )

        print_statement += (
            f"\t{mode.capitalize()} loss: {loss:.4f}"
            f"\tRuntime: {(perf_counter() - t_0):.3f} s"
        )
        print(print_statement)


def load_checkpoint(model, optimizer, checkpoint):
    """Load an existing checkpoint of the model to continue training.

    Params:
        model (torch.nn)             -- Model that should be trained further.
        optimizer (torch.optim)      -- Optimizer that was used.
        checkpoint (torch.load)      -- Checkpoint for continuing to train.
    """
    print("=> Loading checkpoints for critic and generator models")
    # load state dict and optimizer state:
    model.load_state_dict(state_dict=checkpoint["state_dict"])
    optimizer.load_state_dict(state_dict=checkpoint["optimizer"])


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Creates a model checkpoint to save and load a model. The ending <.pth.tar> is commonly used for this.

    Params:
        state (dictionary)      -- The state of the model and optimizer in a dictionary.
        filename (pth.tar)      -- The name of the checkpoint.
    """
    torch.save(state, filename)
    print("\n=> Saving checkpoint")


def count_parameters(model):
    """Calculate the total number of parameters in the model.

    Params:
        model (torch.nn)        -- Model for which we want the total number of parameters.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    return total_params


def check_accuracy(loader, model, mode):
    """
    Check the accuracy of a given model on a given dataset.

    Params:
        loader (torch.utils.data.DataLoader)        -- The dataloader of the
            dataset on which we want to check the accuracy.
        model (torch.nn)                            -- Model for which we want
            the total number of parameters.
        mode (str):                                 -- Mode in which the model
            is in. Either "train" or "test".
    """
    assert mode in ["train", "test"]

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device)
            images = torch.squeeze(
                input=images, dim=1
            )  # shape: ``(batch_size, 28, 28)``, otherwise RNN throws error
            labels = labels.to(device=device)

            forward_pass = model(images)  # shape: ``(batch_size, 10)``
            _, predictions = forward_pass.max(
                dim=1
            )  # from our model, we get the shape ``(batch_size, 10)`` returned
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        print(
            f"{mode.capitalize()} data: Got {num_correct}/{num_samples} with "
            f"accuracy {(100 * num_correct / num_samples):.2f} %"
        )


def produce_loss_plot(num_epochs, train_losses, val_losses, saving_path):
    """Plot the categorical crossentropy (loss) evolving over time.

    Params:
        num_epochs (int)                        -- Number of epochs the model was trained.
        train_losses (numpy.array)              -- Training losses per epoch.
        val_losses (numpy.array)                -- Validation losses per epoch.
        learning_rate (float)                   -- Learning rate for the training of the flow.
        saving_path (str)                       -- Saving path for the loss plot.
    """
    epochs = np.arange(start=0, stop=num_epochs, step=1)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=5.0)
    ax.xaxis.set_major_locator(loc)
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Categorical Crossentropy)")
    plt.legend()
    plt.savefig(
        os.path.join(
            saving_path,
            "loss-lr-" + datetime.now().strftime("%d-%m-%Y-%H:%M") + ".pdf",
        )
    )
    plt.close()


def produce_acc_plot(
    num_epochs, train_accuracies, val_accuracies, saving_path
):
    """Plot the accuracy evolving over time.

    Params:
        num_epochs (int)                        -- Number of epochs the model was trained.
        train_accuracies (numpy.array)          -- Training accuracies per epoch.
        val_accuracies (numpy.array)            -- Validation accuracies per epoch.
        saving_path (str)                       -- Saving path for the loss plot.
    """
    epochs = np.arange(start=0, stop=num_epochs, step=1)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=5.0)
    ax.xaxis.set_major_locator(loc)
    plt.plot(epochs, train_accuracies, label="Training")
    plt.plot(epochs, val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(
            saving_path,
            "accuracy-plot-"
            + datetime.now().strftime("%d-%m-%Y-%H:%M")
            + ".pdf",
        )
    )
    plt.close()


def produce_and_print_confusion_matrix(
    num_classes, test_loader, model, saving_path
):
    """Produce a confusion matrix based on the test set.

    Params:
        num_classes (int)                           -- Number of classes NN has to predict at the end.
        test_loader (torch.utils.data.DataLoader)   -- DataLoader for the test dataset.
        model (torch.nn)                            -- Model that was trained.
        saving_path (str)                           -- Saving path for the loss plot.
    """
    confusion_matrix = torch.zeros(num_classes, num_classes)
    counter = 0
    with torch.no_grad():
        model.eval()
        for i, (inputs, classes) in enumerate(test_loader):
            inputs = inputs.to(device)
            inputs = torch.squeeze(
                input=inputs, dim=1
            )  # shape: (batch_size, 28, 28), otherwise RNN throws error
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t, p] += 1
                counter += 1

    # Because of the random split in the datasets, the classes are imbalanced. Thus, we should do a normalization across each label in the confusion matrix:
    for i in range(num_classes):
        total_sums = 0
        for element in confusion_matrix[i]:
            total_sums += element
        confusion_matrix[i] /= total_sums

    print(f"\nConfusion matrix:\n\n{confusion_matrix}")

    # Convert PyTorch tensor to numpy array:
    fig = plt.figure()
    confusion_matrix = confusion_matrix.detach().cpu().numpy()
    plt.imshow(confusion_matrix, cmap="jet")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(
        os.path.join(
            saving_path,
            "confusion_matrix_"
            + datetime.now().strftime("%d-%m-%Y-%H:%M")
            + ".pdf",
        )
    )

    return confusion_matrix
