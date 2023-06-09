import torch
import torch.nn as nn
from utils.SimpleLayerNorm import LayerNorm


class Conv1dBlock(nn.Module):
    '''
    Description:
    Convolutional layer

    test code:
    conv = Conv1dBlock(1, 32, 7, 1,'lrelu', norm='ln', pad_type='reflect', padding=3)
    x = torch.randn((5,1,1024))
    y = conv(x)
    print(y.size())
    output:
    >> torch.Size([5, 32, 1024])
    '''
    def __init__(self, in_chan, out_chan, kernel_size, stride, activation = 'lrelu', norm='LN', pad_type='reflect', padding=0):
        super(Conv1dBlock,self).__init__()

        self.use_bias = True
        # Padding signal
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad1d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad1d(padding)
        elif pad_type == 'zero':
            self.pad =nn.ConstantPad1d(padding, 0.0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # normalization types
        norm_dim = out_chan
        if norm == 'bn' or norm == 'BN':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in' or norm == 'IN':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln' or norm == 'LN':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm is None:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none' or activation is None:
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # key
        self.conv = nn.Conv1d(in_chan, out_chan, kernel_size, stride, bias=self.use_bias, padding='valid')
        nn.init.kaiming_normal_(self.conv.weight.data, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
