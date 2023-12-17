import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2D, nn.InstanceNorm2d, loss functions etc.
import torch.nn.functional as F  # ReLU activation function, etc.

# Set device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## CREATE AN LSTM:
class LSTM(nn.Module):
    """This class creates an NN for the MNIST dataset. The MNIST dataset has the shape (batch_size, 1, 28, 28)
    when loading it, and we can interpret this shape as 28 sequences, each with 28 features."""

    def __init__(
        self,
        input_size,
        num_layers,
        hidden_size,
        num_classes,
        sequence_length,
        bidirectional,
    ):
        """
        Parameters:
            input_size (int):           In the shape (batch_size, 1, height, width), <height> corresponds to the input size.
            num_layers (int):           Number of hidden layers for the NN.
            hidden_size (int):          Number of features in the hidden state h.
            num_classes (int):          Number of classes our LSTM is supposed to predict, 10 for MNIST.
            sequence_length (int):      Input is of shape (batch_size, sequence_length, input_size).
            bidirectional (boolean):    If <True>, use bidirectional LSTM.
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional

        if self.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # <batch_first = True> because our first axis is the batch_size
            dropout=0,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(
            in_features=self.num_directions
            * self.hidden_size
            * self.sequence_length,
            out_features=self.num_classes,
        )

    def forward(self, x):
        """Standard forward pass."""
        # Initialize hidden state:
        h0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(device)
        # Forward prop:
        out, (hidden_state, cell_state) = self.LSTM(x, (h0, c0))
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
