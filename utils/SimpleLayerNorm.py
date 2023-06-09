import torch
import torch.nn as nn

# pytorch自带的nn.LayerNorm为https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html，但是需要提前指定normalized_shape
#为此，我使用Functional模块对其进行简单的封装，这样无需提前指定normalized_shape，而是在forward阶段从输入的x中进行推导。
# https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html

# class SimpleLayerNorm1d(nn.Module):
#     '''
#     test code
#     x = torch.randn((3,1,1024))
#     B,C,L = x.size()
#     print(B,C,L)

#     the_ln = SimpleLayerNorm1d()
#     y = the_ln(x)

#     print(y.shape)
#     print(list(the_s_ln.parameters()))#这样做的一个问题就是其参数为空，导致无法直接优化


#     '''
#     def __init__(self):
#         super(SimpleLayerNorm1d, self).__init__()

#     def forward(self, x):
#         B, C, L = x.size()
#         y = torch.nn.functional.layer_norm(x, [C,L], eps=1e-05)

#         return y



class LayerNorm(nn.Module):
    '''
    # Copied from DDG code
    # the_ln = LayerNorm(out_chan)
    '''
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
