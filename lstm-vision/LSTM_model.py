import torch
from torch import nn


class LSTM(nn.Module):
    """
    This class creates an NN for the MNIST dataset. The MNIST dataset has
    the shape `(batch_size, 1, 28, 28)`.
    when loading it, and we can interpret this shape as 28 sequences, each with
    28 features.
    """

    def __init__(
        self,
        input_size,
        num_layers,
        hidden_size,
        num_classes,
        sequence_length,
        bidirectional,
        dropout_rate,
        device,
    ):
        """
        Args:
            input_size: input is assumed to be in shape `(N, 1, H, W)`,
                where `W` is the input size
            num_layers: number of hidden layers for the NN
            hidden_size: number of features in hidden state `h`
            num_classes: number of classes our LSTM is supposed to predict,
                `10` for MNIST
            sequence_length: input is of shape
                `(N, sequence_length, input_size)`
            bidirectional: if `True`, use bidirectional LSTM
            dropout_rate: dropout rate for the dropout layer
            device: `cuda` or `cpu`
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.device = device

        if self.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(p=self.dropout_rate, inplace=False)
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
            self.num_layers * self.num_directions,
            x.size(0),
            self.hidden_size,
            device=self.device,
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            x.size(0),
            self.hidden_size,
            device=self.device,
        )

        # Forward prop:
        out, (hidden_state, cell_state) = self.LSTM(x, (h0, c0))
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
