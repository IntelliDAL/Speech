from torch.autograd import Function
import torch.nn as nn
import torch

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss_norm(nn.Module):

    def __init__(self):
        super(DiffLoss_norm, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)

        # Flatten input tensors
        input1 = input1.reshape(batch_size, -1)
        input2 = input2.reshape(batch_size, -1)

        # Calculate mean
        input1_mean = input1.mean(dim=1, keepdim=True)
        input2_mean = input2.mean(dim=1, keepdim=True)

        # Zero-center the data
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        # Compute L2 norms
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True)
        input1_l2 = input1 / (input1_l2_norm + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True)
        input2_l2 = input2 / (input2_l2_norm + 1e-6)

        # Compute the diff_loss
        diff_loss = torch.mean((input1_l2 @ input2_l2.t()).pow(2))


        return diff_loss*10000


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)

        # Flatten input tensors
        input1 = input1.reshape(batch_size, -1)
        input2 = input2.reshape(batch_size, -1)

        # Calculate mean
        input1_mean = input1.mean(dim=1, keepdim=True)
        input2_mean = input2.mean(dim=1, keepdim=True)

        # Zero-center the data
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        # Compute L2 norms
        # input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True)
        # input1_l2 = input1 / (input1_l2_norm + 1e-6)
        #
        # input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True)
        # input2_l2 = input2 / (input2_l2_norm + 1e-6)

        # Compute the diff_loss
        diff_loss = torch.mean((input1 @ input2.t()).pow(2))

        return diff_loss*1000


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms * 0.01

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
