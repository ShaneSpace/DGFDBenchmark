import torch

class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd)
    return GradReverse.apply(x, lam)
