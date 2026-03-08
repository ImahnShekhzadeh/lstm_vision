import torch
from typeguard import typechecked

from lstm_vision.model import LSTM


@typechecked
def test__model_output() -> None:
    """Test the output shape of the LSTM."""

    num_classes = 2
    model = LSTM(
        input_size=32,
        sequence_length=28,
        num_layers=3,
        hidden_size=256,
        num_classes=num_classes,
        bidirectional=True,
        dropout_rate=0.0,
    )
    x = torch.rand(1024, 28, 32)
    assert model(x).shape == torch.Size([1024, num_classes])
