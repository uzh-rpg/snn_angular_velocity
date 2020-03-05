import warnings
import torch
from torch.nn.utils.rnn import PackedSequence


def moveToGPUDevice(
        input,
        device: torch.device,
        dtype: torch.dtype = None,
        non_blocking: bool = True):
    if torch.cuda.is_available():
        if isinstance(input, torch.Tensor):
            return input.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if isinstance(input, torch.nn.Module):
            # Performs in-place modification of the module but we still return for convenience.
            return input.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if isinstance(input, PackedSequence):
            # NOTE: PackedSequence is also a tuple, apparently! However, we should directly move it using to().
            return input.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if isinstance(input, tuple):
            return tuple(moveToGPUDevice(v, device=device, dtype=dtype, non_blocking=non_blocking)
                         for v in input)
        if isinstance(input, dict):
            return {k: moveToGPUDevice(v, device=device, dtype=dtype, non_blocking=non_blocking)
                    for k, v in input.items()}
        if isinstance(input, list):
            return [moveToGPUDevice(v, device=device, dtype=dtype, non_blocking=non_blocking) for v in input]
        warnings.warn("Unknown instance! Input remains on current device!", Warning)
        return input
    warnings.warn("Cuda not available! Input remains on CPU!", Warning)
    return input
