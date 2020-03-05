import torch


class SpikeRepresentationGenerator:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.num_time_bins = None

    def getSlayerSpikeTensor(self,
                             ev_pol: torch.Tensor,
                             ev_xy: torch.Tensor,
                             ev_ts_us: torch.Tensor,
                             ang_ts_us: torch.Tensor):
        if self.num_time_bins is None:
            self.num_time_bins = ang_ts_us.numel()
        else:
            assert self.num_time_bins == ang_ts_us.numel()

        ang_ts_diff = ang_ts_us[1:] - ang_ts_us[:-1]
        ts_diff_us = ang_ts_diff[0]
        assert torch.all(ang_ts_diff[0] == ang_ts_diff)

        # The following element-wise division is performed by integer tensors which will floor the values as intended.
        time_idx = ev_ts_us/ts_diff_us
        # Handle time stamps that are not floored and would exceed the allowed index after to-index conversion.
        time_idx[time_idx == self.num_time_bins] = self.num_time_bins - 1

        spike_tensor = torch.zeros((2, self.height, self.width, self.num_time_bins))
        spike_tensor[ev_pol.long(), ev_xy[:, 1].long(), ev_xy[:, 0].long(), time_idx.long()] = 1
        return spike_tensor
