from .base import (
    DataType,
    MetaTensor,
    SpikeModule,
    TensorLayout,
    getNeuronConfig
)
import slayerpytorch as snn
import torch


class SNN(SpikeModule):
    def __init__(self, simulation_params):
        super().__init__()

        conv1_out_channels = 16
        conv1_kernelsize = 3
        conv1_stride = 2
        conv1_padding = 0
        conv1_dilation = 1
        tauSr1 = 2.
        tauRef1 = 1.
        neuron_config_conv1 = getNeuronConfig(theta=0.27,
                                              tauSr=tauSr1,
                                              tauRef=tauRef1,
                                              scaleRho=0.20)
        self.slayer_conv1 = snn.layer(neuron_config_conv1, simulation_params)
        self.conv1 = self.slayer_conv1.conv(inChannels=2,
                                            outChannels=conv1_out_channels,
                                            kernelSize=conv1_kernelsize,
                                            stride=conv1_stride,
                                            padding=conv1_padding,
                                            dilation=conv1_dilation,
                                            groups=1,
                                            weightScale=1)
        conv2_out_channels = 32
        conv2_kernelsize = 3
        conv2_stride = 2
        conv2_padding = 0
        conv2_dilation = 1
        tauSr2 = 2.
        tauRef2 = 1.
        neuron_config_conv2 = getNeuronConfig(theta=0.25,
                                              tauSr=tauSr2,
                                              tauRef=tauRef2,
                                              scaleRho=0.13)
        self.slayer_conv2 = snn.layer(neuron_config_conv2, simulation_params)
        self.conv2 = self.slayer_conv2.conv(inChannels=conv1_out_channels,
                                            outChannels=conv2_out_channels,
                                            kernelSize=conv2_kernelsize,
                                            stride=conv2_stride,
                                            padding=conv2_padding,
                                            dilation=conv2_dilation,
                                            groups=1,
                                            weightScale=1)

        conv3_out_channels = 64
        conv3_kernelsize = 3
        conv3_stride = 2
        conv3_padding = 0
        conv3_dilation = 1
        tauSr3 = 4.
        tauRef3 = 4.
        neuron_config_conv3 = getNeuronConfig(theta=0.3,
                                              tauSr=tauSr3,
                                              tauRef=tauRef3,
                                              scaleRho=0.13)
        self.slayer_conv3 = snn.layer(neuron_config_conv3, simulation_params)
        self.conv3 = self.slayer_conv3.conv(inChannels=conv2_out_channels,
                                            outChannels=conv3_out_channels,
                                            kernelSize=conv3_kernelsize,
                                            stride=conv3_stride,
                                            padding=conv3_padding,
                                            dilation=conv3_dilation,
                                            groups=1,
                                            weightScale=1)
        conv4_out_channels = 128
        conv4_kernelsize = 3
        conv4_stride = 2
        conv4_padding = 0
        conv4_dilation = 1
        tauSr4 = 4.
        tauRef4 = 4.
        neuron_config_conv4 = getNeuronConfig(theta=0.4,
                                              tauSr=tauSr4,
                                              tauRef=tauRef4,
                                              scaleRho=0.25)
        self.slayer_conv4 = snn.layer(neuron_config_conv4, simulation_params)
        self.conv4 = self.slayer_conv4.conv(inChannels=conv3_out_channels,
                                            outChannels=conv4_out_channels,
                                            kernelSize=conv4_kernelsize,
                                            stride=conv4_stride,
                                            padding=conv4_padding,
                                            dilation=conv4_dilation,
                                            groups=1,
                                            weightScale=1)
        conv5_out_channels = 256
        conv5_kernelsize = 3
        conv5_stride = 1
        conv5_padding = 0
        conv5_dilation = 1
        tauSr5 = 4.
        tauRef5 = 4.
        neuron_config_conv5 = getNeuronConfig(theta=0.4,
                                              tauSr=tauSr5,
                                              tauRef=tauRef5,
                                              scaleRho=100.)
        self.slayer_conv5 = snn.layer(neuron_config_conv5, simulation_params)
        self.conv5 = self.slayer_conv5.conv(inChannels=conv4_out_channels,
                                            outChannels=conv5_out_channels,
                                            kernelSize=conv5_kernelsize,
                                            stride=conv5_stride,
                                            padding=conv5_padding,
                                            dilation=conv5_dilation,
                                            groups=1,
                                            weightScale=1)

        # NOTE: tauSr is the only parameter that has any influence on the output or the gradients.
        neuron_config_fc = getNeuronConfig(tauSr=8.0)
        self.slayer_fc = snn.layer(neuron_config_fc, simulation_params)
        self.fc = self.slayer_fc.dense(inFeatures=conv5_out_channels,
                                       outFeatures=3,
                                       weightScale=1)

    def forward(self, spike_input):
        self.addInputMetaTensor(MetaTensor(spike_input, TensorLayout.Conv, DataType.Spike))

        spikes_layer_1 = self.slayer_conv1.spike(self.conv1(self.slayer_conv1.psp(spike_input)))
        spikes_layer_2 = self.slayer_conv2.spike(self.conv2(self.slayer_conv2.psp(spikes_layer_1)))
        spikes_layer_3 = self.slayer_conv3.spike(self.conv3(self.slayer_conv3.psp(spikes_layer_2)))
        spikes_layer_4 = self.slayer_conv4.spike(self.conv4(self.slayer_conv4.psp(spikes_layer_3)))
        spikes_layer_5 = self.slayer_conv5.spike(self.conv5(self.slayer_conv5.psp(spikes_layer_4)))

        self.addMetaTensor('conv1', MetaTensor(spikes_layer_1, TensorLayout.Conv, DataType.Spike))
        self.addMetaTensor('conv2', MetaTensor(spikes_layer_2, TensorLayout.Conv, DataType.Spike))
        self.addMetaTensor('conv3', MetaTensor(spikes_layer_3, TensorLayout.Conv, DataType.Spike))
        self.addMetaTensor('conv4', MetaTensor(spikes_layer_4, TensorLayout.Conv, DataType.Spike))
        self.addMetaTensor('conv5', MetaTensor(spikes_layer_5, TensorLayout.Conv, DataType.Spike))

        # Apply average pooling on spike-trains.
        spikes_mean = torch.mean(spikes_layer_5, dim=(2, 3), keepdims=True)
        psp_out = self.slayer_fc.psp(self.fc(spikes_mean))
        self.addOutputMetaTensor(MetaTensor(psp_out, TensorLayout.FC, DataType.Dense))

        assert psp_out.shape[1] == 3
        # Return: (batch, ang-vel, time)
        return psp_out.squeeze()
