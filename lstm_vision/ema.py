from torch import nn


def update_ema_model(
    model: nn.Module, ema_model: nn.Module, momentum: float
) -> None:
    """
    Update the EMA (exponential moving average) model according to the formula
    `xi_{t+1} = rho * xi_t + (1 - rho) * theta_t,`
    where `rho` is the momentum, `theta_t` are the model parameters and
    `xi_t` are the EMA model parameters. Also see https://arxiv.org/pdf/2307.13813. Implementation inspired by https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/utils_cifar.py#L76-L82

    Args:
        model: Model.
        ema_model: EMA model.
        momentum: Momentum.
    """

    model__state_dict = model.state_dict()
    ema_model__state_dict = ema_model.state_dict()

    for key in model__state_dict:
        ema_model__state_dict[key].copy_(
            ema_model__state_dict[key] * momentum
            + model__state_dict[key] * (1 - momentum)
        )
