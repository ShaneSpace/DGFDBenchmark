import random
import torch
import torch.nn as nn
# Author: Jia Linshan
# 2022-10-31, Monday
# The form of x should be [Channel,Length], because it is used in the 'Dataset.__getitem__()' directly.
class AddGaussianNoise(nn.Module):
    def __init__(self, sigma=0.05):
        super(AddGaussianNoise,self).__init__()
        self.sigma = sigma

    def forward(self,x):
        return x + torch.randn(x.size())*self.sigma

class RandomScale(nn.Module):
    def __init__(self, sigma=0.05):
        super(RandomScale, self).__init__()
        if sigma<=0:
            raise ValueError('The sigma should be no less than zero!')
        self.sigma = sigma
    def forward(self, x):
        s = torch.randn(size=(1,))*self.sigma+1
        return s*x

class MakeNoise(nn.Module):
    def __init__(self, mu=0.1):
        super(MakeNoise,self).__init__()
        self.mu = mu

    def forward(self,x):
        mask_float = torch.rand(size=x.size())
        zeros = torch.zeros(size=x.size())
        ones = torch.ones(size = x.size())
        mask = torch.where(mask_float<self.mu, zeros, ones)
        return torch.mul(x, mask)

class Translation(nn.Module):
    def __init__(self):
        super(Translation, self).__init__()
    def forward(self, x):
        length_x = len(x)
        cut_point_01  = torch.rand((1,))
        cut_point = torch.floor(length_x*cut_point_01).type(torch.int64)
        y = torch.cat((x[cut_point:], x[:cut_point]))

        return y

class SignalInverse(nn.Module):
    def __init__(self):
        super(SignalInverse, self).__init__()
    def forward(self,x):
        # we assume the x is zero-mean form
        return -x

class TimeScretch(nn.Module):
    def __init__(self, sigma=0.05):
        super(TimeScretch, self).__init__()
        self.sigma = sigma
    def forward(self,x):
        x = torch.unsqueeze(x, dim=0)
        stretch_factor = torch.randn((1,))*self.sigma+1
        y = nn.Upsample(scale_factor = stretch_factor)(x)
        if y.size(2)>x.size(2):
            y = y[:,:,:x.size(2)]
        elif y.size(2)<x.size(2):
            print(torch.zeros((y.size(0), y.size(1), x.size(2)-y.size(2))).shape)
            print(y.size(2))
            y = torch.cat((y, torch.zeros((y.size(0), y.size(1), x.size(2)-y.size(2)))),dim=2)

        return y


