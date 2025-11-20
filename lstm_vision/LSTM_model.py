import torch
from torch import nn
from typeguard import typechecked


@typechecked
class LSTM(nn.Module):
    """
    This class creates an NN for the MNIST dataset. The MNIST dataset has
    the shape `(batch_size, 1, 28, 28)`.
    when loading it, and we can interpret this shape as 28 sequences, each with
    28 features.
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        num_layers: int,
        hidden_size: int,
        num_classes: int,
        bidirectional: bool,
        dropout_rate: float,
    ) -> None:
        """
        Args:
            input_size: input is assumed to be in shape `(N, 1, H, W)`,
                where `W` is the input size
            sequence_length: input is of shape
                `(N, sequence_length, input_size)`
            num_layers: number of hidden layers for the NN
            hidden_size: number of features in hidden state `h`
            num_classes: number of classes our LSTM is supposed to predict,
                `10` for MNIST
            bidirectional: if `True`, use bidirectional LSTM
            dropout_rate: dropout rate for the dropout layer
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)
        self.fc = nn.Linear(
            in_features=self.num_directions
            * self.hidden_size
            * sequence_length,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Args:
            x: Input tensor to be passed through the LSTM.

        Returns:
            Output of the LSTM, reshaped.
        """
        out, (_, _) = self.LSTM(x)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out
