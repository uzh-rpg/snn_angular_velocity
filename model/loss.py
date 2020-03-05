import torch


def compute_loss(input_: torch.Tensor, target: torch.Tensor, time_start_idx: int=None):
    """
    :param input: tensor of shape (batch, 3, time)
    :param target: tensor of shape (batch, 3, time)
    :param time_start_idx: Time-index from which to start computing the loss
    :return: loss
    """
    assert len(input_.shape) == 3
    assert len(target.shape) == 3
    assert input_.shape == target.shape
    assert input_.shape[1] == 3

    if time_start_idx:
        input_ = input_[..., time_start_idx:]
        target = target[..., time_start_idx:]

    return torch.mean(torch.sqrt(torch.sum((input_ - target) ** 2, dim=1)))
