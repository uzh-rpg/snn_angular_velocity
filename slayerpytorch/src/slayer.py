import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import slayerCuda


class spikeLayer(torch.nn.Module):
    '''
    This class defines the main engine of SLAYER.
    It provides necessary functions for describing a SNN layer.
    The input to output connection can be fully-connected, convolutional, or aggregation (pool)
    It also defines the psp operation and spiking mechanism of a spiking neuron in the layer.

    **Important:** It assumes all the tensors that are being processed are 5 dimensional. 
    (Batch, Channels, Height, Width, Time) or ``NCHWT`` format.
    The user must make sure that an input of correct dimension is supplied.

    *If the layer does not have spatial dimension, the neurons can be distributed along either
    Channel, Height or Width dimension where Channel * Height * Width is equal to number of neurons.
    It is recommended (for speed reasons) to define the neuons in Channels dimension and make Height and Width
    dimension one.*

    Arguments:
        * ``neuronDesc`` (``slayerParams.yamlParams``): spiking neuron descriptor.
            .. code-block:: python

                neuron:
                    type:     SRMALPHA  # neuron type
                    theta:    10    # neuron threshold
                    tauSr:    10.0  # neuron time constant
                    tauRef:   1.0   # neuron refractory time constant
                    scaleRef: 2     # neuron refractory response scaling (relative to theta)
                    tauRho:   1     # spike function derivative time constant (relative to theta)
                    scaleRho: 1     # spike function derivative scale factor
        * ``simulationDesc`` (``slayerParams.yamlParams``): simulation descriptor
            .. code-block:: python

                simulation:
                    Ts: 1.0         # sampling time (ms)
                    tSample: 300    # time length of sample (ms)   
        * ``fullRefKernel`` (``bool``, optional): high resolution refractory kernel (the user shall not use it in practice)  

    Usage:

    >>> snnLayer = slayer.spikeLayer(neuronDesc, simulationDesc)
    '''
    def __init__(self, neuronDesc, simulationDesc, fullRefKernel = False):
        super(spikeLayer, self).__init__()
        self.neuron = neuronDesc
        self.simulation = simulationDesc
        self.fullRefKernel = fullRefKernel

        self.register_buffer('srmKernel', self.calculateSrmKernel())
        self.register_buffer('refKernel', self.calculateCustomRefKernel())

    def calculateSrmKernel(self):
        srmKernel = self._calculateAlphaKernel(self.neuron['tauSr'])
        return torch.tensor(srmKernel)

    def calculateRefKernel(self):
        if self.fullRefKernel:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'], EPSILON = 0.0001)
            # This gives the high precision refractory kernel as MATLAB implementation, however, it is expensive
        else:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'])

        return torch.tensor(refKernel)

    def calculateCustomRefKernel(self, EPSILON: float = 0.01):
        tau = self.neuron['tauRef']
        mult = - self.neuron['scaleRef'] * self.neuron['theta']
        assert tau > 0.
        assert mult < 0.
        #print('Reference kernel with tau = {} crosses negative threshold at {}'.format( tau, - tau * np.log(1 / 2) + self.simulation['Ts']))
        time = np.arange(0, self.simulation['tSample'], self.simulation['Ts'])
        potential = mult * np.exp(-1 / tau * time[:-1])
        return torch.from_numpy(np.concatenate((np.array([0]), potential[np.abs(potential) > EPSILON]))).float()

    def _calculateAlphaKernel(self, tau, mult = 1, EPSILON = 0.01):
        eps = []
        for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
            epsVal = mult * t / tau * math.exp(1 - t / tau)
            if abs(epsVal) < EPSILON and t > tau:
                break
            eps.append(epsVal)
        return eps

    def _zeroPadAndFlip(self, kernel):
        if (len(kernel)%2) == 0: kernel.append(0)
        prependedZeros = np.zeros((len(kernel) - 1))
        return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()

    def psp(self, spike):
        '''
        Applies psp filtering to spikes.
        The output tensor dimension is same as input.

        Arguments:
            * ``spike``: input spike tensor.

        Usage:

        >>> filteredSpike = snnLayer.psp(spike)
        '''
        return _pspFunction.apply(spike, self.srmKernel, self.simulation['Ts'])

    def pspLayer(self):
        '''
        Returns a function that can be called to apply psp filtering to spikes.
        The output tensor dimension is same as input.
        The initial psp filter corresponds to the neuron psp filter.
        The psp filter is learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.

        Usage:

        >>> pspLayer = snnLayer.pspLayer()
        >>> filteredSpike = pspLayer(spike)
        '''
        return _pspLayer(self.srmKernel, self.simulation['Ts'])

    def pspFilter(self, nFilter, filterLength, filterScale=1):
        '''
        Returns a function that can be called to apply a bank of temporal filters.
        The output tensor is of same dimension as input except the channel dimension is scaled by number of filters.
        The initial filters are initialized using default PyTorch initializaion for conv layer.
        The filter banks are learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.

        Arguments:
            * ``nFilter``: number of filters in the filterbank.
            * ``filterLength``: length of filter in number of time bins.
            * ``filterScale``: initial scaling factor for filter banks. Default: 1.

        Usage:

        >>> pspFilter = snnLayer.pspFilter()
        >>> filteredSpike = pspFilter(spike)
        '''
        return _pspFilter(nFilter, filterLength, self.simulation['Ts'], filterScale)

    def replicateInTime(self, input, mode='nearest'):
        Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
        N, C, H, W = input.shape
        # output = F.pad(input.reshape(N, C, H, W, 1), pad=(Ns-1, 0, 0, 0, 0, 0), mode='replicate')
        if mode == 'nearest':
            output = F.interpolate(input.reshape(N, C, H, W, 1), size=(H, W, Ns), mode='nearest')
        return output

    def dense(self, inFeatures, outFeatures, weightScale=10):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10

        Usage:

        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _denseLayer(inFeatures, outFeatures, weightScale)    

    def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100):    # default weight scaling of 100
        '''
        Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.conv2d`` applied for each time instance.

        Arguments:
            * ``inChannels`` (``int``): number of channels in input
            * ``outChannels`` (``int``): number of channls produced by convoluion
            * ``kernelSize`` (``int`` or tuple of two ints): size of the convolving kernel
            * ``stride`` (``int`` or tuple of two ints): stride of the convolution. Default: 1
            * ``padding`` (``int`` or tuple of two ints):   zero-padding added to both sides of the input. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
            * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
            * ``weightScale``: sale factor of default initialized weights. Default: 100

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> conv = snnLayer.conv(2, 32, 5) # 32C5 flter
        >>> output = conv(input)           # must have 2 channels
        '''
        return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale) 

    def pool(self, kernelSize, stride=None, padding=0, dilation=1):
        '''
        Returns a function that can be called to apply pool layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.``:sum pooling applied for each time instance.

        Arguments:
            * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to pool over
            * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
            * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> pool = snnLayer.pool(4) # 4x4 pooling
        >>> output = pool(input)
        '''
        return _poolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation)

    def dropout(self, p=0.5, inplace=False):
        '''
        Returns a function that can be called to apply dropout layer to the input tensor.
        It behaves similar to ``torch.nn.Dropout``.
        However, dropout over time dimension is preserved, i.e.
        if a neuron is dropped, it remains dropped for entire time duration.

        Arguments:
            * ``p``: dropout probability.
            * ``inplace`` (``bool``): inplace opeartion flag.

        Usage:

        >>> drop = snnLayer.dropout(0.2)
        >>> output = drop(input)
        '''
        return _dropoutLayer(p, inplace)

    def delayShift(self, input, delay, Ts=1):
        '''
        Applies delay in time dimension (assumed to be the last dimension of the tensor) of the input tensor.
        The autograd backward link is established as well.

        Arguments:
            * ``input``: input Torch tensor.
            * ``delay`` (``float`` or Torch tensor): amount of delay to apply.
              Same delay is applied to all the inputs if ``delay`` is ``float`` or Torch tensor of size 1.
              If the Torch tensor has size more than 1, its dimension  must match the dimension of input tensor except the last dimension.
            * ``Ts``: sampling time of the delay. Default is 1.

        Usage:

        >>> delayedInput = slayer.delayShift(input, 5)
        '''
        return _delayFunctionNoGradient.apply(input, delay, Ts)

    def delay(self, inputSize):
        '''
        Returns a function that can be called to apply delay opeartion in time dimension of the input tensor.
        The delay parameter is available as ``delay.delay`` and is initialized uniformly between 0ms  and 1ms.
        The delay parameter is stored as float values, however, it is floored during actual delay applicaiton internally.
        The delay values are not clamped to zero.
        To maintain the causality of the network, one should clamp the delay values explicitly to ensure positive delays.

        Arguments:
            * ``inputSize`` (``int`` or tuple of three ints): spatial shape of the input signal in CHW format (Channel, Height, Width).
              If integer value is supplied, it refers to the number of neurons in channel dimension. Heighe and Width are assumed to be 1.   

        Usage:

        >>> delay = snnLayer.delay((C, H, W))
        >>> delayedSignal = delay(input)

        Always clamp the delay after ``optimizer.step()``.

        >>> optimizer.step()
        >>> delay.delay.data.clamp_(0)  
        '''
        return _delayLayer(inputSize, self.simulation['Ts'])

    def spike(self, membranePotential):
        '''
        Applies spike function and refractory response.
        The output tensor dimension is same as input.
        ``membranePotential`` will reflect spike and refractory behaviour as well.

        Arguments:
            * ``membranePotential``: subthreshold membrane potential.

        Usage:

        >>> outSpike = snnLayer.spike(membranePotential)
        '''
        return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])


class _denseLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale=1):
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed

    def forward(self, input):
        return F.conv3d(input, 
                        self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class _convLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1):
        inChannels = inFeatures
        outChannels = outFeatures

        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(_convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed

    def foward(self, input):
        return F.conv3d(input, 
                        self.weight, self.bias, 
                        self.stride, self.padding, self.dilation, self.groups)


class _poolLayer(nn.Conv3d):
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1):
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))
        super(_poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)   

        # set the weights to 1.1*theta and requires_grad = False
        self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)


    def forward(self, input):
        device = input.device
        dtype  = input.dtype

        if input.shape[2]%self.weight.shape[2] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        if input.shape[3]%self.weight.shape[3] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]), dtype=dtype).to(device)), 3)

        dataShape = input.shape

        result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                          self.weight, self.bias, 
                          self.stride, self.padding, self.dilation)

        return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))


class _dropoutLayer(nn.Dropout3d):
    def forward(self, input):
        inputShape = input.shape
        return F.dropout3d(input.reshape((inputShape[0], -1, 1, 1, inputShape[-1])),
                           self.p, self.training, self.inplace).reshape(inputShape)


class _pspLayer(nn.Conv3d):
    def __init__(self, filter, Ts):
        inChannels  = 1
        outChannels = 1
        kernel      = (1, 1, torch.numel(filter))

        self.Ts = Ts

        super(_pspLayer, self).__init__(inChannels, outChannels, kernel, bias=False) 

        flippedFilter = torch.tensor(np.flip(filter.cpu().data.numpy()).copy()).reshape(self.weight.shape)

        self.weight = torch.nn.Parameter(flippedFilter.to(self.weight.device), requires_grad = True)

        self.pad = torch.nn.ConstantPad3d(padding=(torch.numel(filter)-1, 0, 0, 0, 0, 0), value=0)

    def forward(self, input):
        inShape = input.shape
        inPadded = self.pad(input.reshape((inShape[0], 1, 1, -1, inShape[-1])))
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape(inShape)


class _pspFilter(nn.Conv3d):
    def __init__(self, nFilter, filterLength, Ts, filterScale=1):
        inChannels  = 1
        outChannels = nFilter
        kernel      = (1, 1, filterLength)

        super(_pspFilter, self).__init__(inChannels, outChannels, kernel, bias=False) 

        self.Ts  = Ts
        self.pad = torch.nn.ConstantPad3d(padding=(filterLength-1, 0, 0, 0, 0, 0), value=0)

        if filterScale != 1:
            self.weight.data *= filterScale

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        inPadded = self.pad(input.reshape((N, 1, 1, -1, Ns)))
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape((N, -1, H, W, Ns))


class _spikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
        device = membranePotential.device
        dtype  = membranePotential.dtype
        threshold      = neuron['theta']
        oldDevice = torch.cuda.current_device()
        spikes = slayerCuda.getSpikes(membranePotential, refractoryResponse, threshold, Ts)

        pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
        pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        (membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
        reasonable = True
        if reasonable:
            # For some reason the membrane potential clamping does not give good results.
            # membranePotential[membranePotential > threshold] = threshold
            # For some reason pdfScale should be scaled by the threshold to get decent results.
            spikePdf = pdfScale/threshold * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)
        else:
            spikePdf = pdfScale/pdfTimeConstant * torch.exp( -torch.abs(membranePotential - threshold) / pdfTimeConstant)

        return gradOutput * spikePdf, None, None, None


class _pspFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike, filter, Ts):
        device = spike.device
        dtype  = spike.dtype
        psp = slayerCuda.conv(spike.contiguous(), filter, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(filter, Ts)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        (filter, Ts) = ctx.saved_tensors
        gradInput = slayerCuda.corr(gradOutput.contiguous(), filter, Ts)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass
        return gradInput, gradFilter, None


class _delayLayer(nn.Module):
    def __init__(self, inputSize, Ts):
        super(_delayLayer, self).__init__()

        if type(inputSize) == int:
            inputChannels = inputSize
            inputHeight   = 1
            inputWidth    = 1
        elif len(inputSize) == 3:
            inputChannels = inputSize[0]
            inputHeight   = inputSize[1]
            inputWidth    = inputSize[2]
        else:
            raise Exception('inputSize can only be 1 or 2 dimension. It was: {}'.format(inputSize.shape))

        self.delay = torch.nn.Parameter(torch.rand((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        self.Ts = Ts

    def forward(self, input):
        N, C, H, W, Ns = input.shape
        if input.numel() != self.delay.numel() * input.shape[-1]:
            return _delayFunction.apply(input, self.delay.repeat((1, H, W)), self.Ts) # different delay per channel
        else:
            return _delayFunction.apply(input, self.delay, self.Ts)


class _delayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delay, Ts):
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input, delay.data, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(output, delay.data, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        (output, delay, Ts) = ctx.saved_tensors
        diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
        outputDiff = slayerCuda.conv(output, diffFilter, 1)
        # the conv operation should not be scaled by Ts. 
        # As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
        gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
        # no minus needed here, as it is included in diffFilter which is -1 * [1, -1]

        return slayerCuda.shift(gradOutput, -delay, Ts), gradDelay, None


class _delayFunctionNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delay, Ts=1):
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input, delay, Ts)
        Ts     = torch.autograd.Variable(torch.tensor(Ts   , device=device, dtype=dtype), requires_grad=False)
        delay  = torch.autograd.Variable(torch.tensor(delay, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(delay, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        (delay, Ts) = ctx.saved_tensors
        return slayerCuda.shift(gradOutput, -delay, Ts), None, None
