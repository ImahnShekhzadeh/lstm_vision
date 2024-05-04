import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import autocast, nn
from torch.utils.data import DataLoader


@torch.no_grad()
def check_accuracy(
    loader: DataLoader,
    model: nn.Module,
    use_amp: bool,
    mode: str,
    device: int | torch.device,
) -> None:
    """
    Check the accuracy of a given model on a given dataset.

    Args:
        loader: Dataloader of the dataset for which to check the accuracy.
        model: Model.
        use_amp: Whether to use automatic mixed precision.
        mode: Mode in which the model is in. Either "train" or "test".
        device: Device on which the code is executed.
    """
    assert mode in ["train", "test"]

    model.eval()
    num_correct, num_samples = 0, 0

    for images, labels in loader:
        labels = labels.to(device)
        with autocast(
            device_type=labels.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            forward_pass = model(images.squeeze(dim=1).to(device))  # `(N, 10)`

        predictions = forward_pass.argmax(dim=1)
        num_correct += (predictions == labels).sum().cpu().item()
        num_samples += predictions.shape[0]

    logging.info(
        f"{mode.capitalize()} data: Got {num_correct}/{num_samples} with "
        f"accuracy {(100 * num_correct / num_samples):.2f} %"
    )


@torch.no_grad()
def get_confusion_matrix(
    num_classes: int,
    test_loader: DataLoader,
    model: nn.Module,
    use_amp: bool,
    saving_path: str,
    device: int | torch.device,
    timestamp: str,
) -> np.ndarray:
    """
    Produce a confusion matrix based on the test set.

    Args:
        num_classes: Number of classes neural networks predicts at the end.
        test_loader: DataLoader for the test dataset.
        model: Model.
        use_amp: Whether to use automatic mixed precision.
        saving_path: Saving path where the confusion matrix is stored.
        device: Device on which the code is executed.
        timestamp: Timestamp of the current run.

    Returns:
        Confusion matrix.
    """
    model.eval()
    counter = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for i, (inputs, classes) in enumerate(test_loader):
        classes = classes.to(device)

        with autocast(
            device_type=classes.device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(inputs.squeeze(dim=1).to(device))

        preds = outputs.argmax(dim=1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t, p] += 1
            counter += 1

    # Because of the random split in the datasets, the classes are imbalanced.
    # Thus, we should do a normalization across each label in the confusion
    # matrix:
    for i in range(num_classes):
        total_sums = 0
        for element in confusion_matrix[i]:
            total_sums += element
        confusion_matrix[i] /= total_sums

    logging.info(f"\nConfusion matrix:\n\n{confusion_matrix}")

    # Convert PyTorch tensor to numpy array:
    fig = plt.figure()
    confusion_matrix = confusion_matrix.numpy()
    font = {"size": 7}
    for (i, j), label in np.ndenumerate(confusion_matrix):
        plt.text(j, i, f"{label:.3f}", ha="center", va="center", fontdict=font)
        plt.text(j, i, f"{label:.3f}", ha="center", va="center", fontdict=font)
    tick_marks = np.arange(start=0, stop=num_classes)
    plt.xticks(ticks=tick_marks)
    plt.yticks(ticks=tick_marks)
    plt.imshow(confusion_matrix, cmap="jet")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(
        os.path.join(
            saving_path,
            f"confusion_matrix_{timestamp}.pdf",
        ),
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )

    return confusion_matrix
