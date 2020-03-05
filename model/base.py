from enum import Enum, auto
import torch
from typing import Tuple
import warnings


class TensorLayout(Enum):
    Conv = auto()
    FC = auto()


class DataType(Enum):
    Spike = auto()
    Dense = auto()


class MetaTensor:
    def __init__(self,
                 tensor: torch.Tensor,
                 tensor_layout: TensorLayout,
                 data_type: DataType):
        self._data = tensor
        self._tensor_layout = tensor_layout
        self._data_type = data_type

        if self.hasFCLayout():
            assert tensor.shape[2:4] == (1, 1)

    def getTensor(self):
        return self._data.detach()

    def getMeanNumSpikesPerNeuron(self):
        """
        :return: For each sample in the batch, the average number of spikes per neuron in the given time.
                 Returns None if data does not contain spikes.
        """
        if not self.isSpikeType():
            warnings.warn("Data does not contain spikes. Returning None.", Warning)
            return None
        if self.hasConvLayout():
            return torch.mean(self._data.sum(4), dim=(1, 2, 3)).detach()
        assert self.hasFCLayout()
        return torch.mean(self._data.squeeze().sum(2), dim=1).detach()

    def getSpikeCounts(self):
        """
        :return: For each sample in the batch, return the number of spikes, number of neurons and number of timesteps.
                 Returns None if data does not contain spikes.
        """
        output = dict()
        data = self._data.detach()
        if not self.isSpikeType():
            warnings.warn("Data does not contain spikes. Returning None.", Warning)
            return None
        if self.hasConvLayout():
            num_spikes = torch.sum(data, dim=(1, 2, 3, 4))
            num_neurons = data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4]
            num_steps = data.shape[4]
        else:
            assert self.hasFCLayout()
            data = data.squeeze()  # (batch, neurons, time)
            num_spikes = torch.sum(data, dim=(1, 2)).detach()
            num_neurons = data.shape[1]
            num_steps = data.shape[2]
        output['num_spikes'] = num_spikes
        output['num_neurons'] = num_neurons
        output['num_steps'] = num_steps

        num_spikes_per_neuron = torch.sum(data, dim=-1).long()
        is_spiking = num_spikes_per_neuron >= 1
        num_steps_per_spike_per_neuron = num_steps / num_spikes_per_neuron[is_spiking].float()

        fraction_spiking = torch.sum(is_spiking).float() / num_spikes_per_neuron.numel()

        # steps/(spikes/neuron) as flattened vector over the whole batch:
        output['fraction_spiking'] = fraction_spiking.item()
        # spikes/neuron as flattened vector over the whole batch (only includes spiking neurons):
        output['spikes_per_neuron'] = num_spikes_per_neuron[is_spiking]
        # steps/(spikes/neuron) as flattened vector over the whole batch (only includes spiking neurons):
        output['steps_in_batch'] = num_steps_per_spike_per_neuron

        return output

    def isSpikeType(self):
        return self._data_type == DataType.Spike

    def isDenseType(self):
        return self._data_type == DataType.Dense

    def hasConvLayout(self):
        return self._tensor_layout == TensorLayout.Conv

    def hasFCLayout(self):
        return self._tensor_layout == TensorLayout.FC


class SpikeModule(torch.nn.Module):
    _input_key = 'input'
    _output_key = 'output'

    def __init__(self):
        super().__init__()
        self._data = dict()

    def getMetaTensorDict(self):
        return self._data

    def addMetaTensor(self, key: str, value: MetaTensor):
        assert not key == self._output_key, 'Use addOutputMetaTensor function instead'
        self._data[key] = value

    def addInputMetaTensor(self, value: MetaTensor):
        assert value.isSpikeType()
        assert value.hasConvLayout(), 'Does not have to be but is reasonable for the moment'
        self._data[self._input_key] = value

    def addOutputMetaTensor(self, value: MetaTensor):
        assert value.isDenseType()
        assert value.hasFCLayout(), 'Does not have to be but is reasonable for the moment'
        self._data[self._output_key] = value


def getNeuronConfig(type: str='SRMALPHA',
                    theta: float=10.,
                    tauSr: float=1.,
                    tauRef: float=1.,
                    scaleRef: float=2.,
                    tauRho: float=0.3,  # Was set to 0.2 previously (e.g. for fullRes run)
                    scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }
