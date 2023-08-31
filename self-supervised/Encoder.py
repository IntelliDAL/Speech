from torch import nn
from model.model_main import supervisedBackbone as supBackOne
from model.model_main import supervisedBackbTwo as supBackTwo


class EncoderIT(nn.Module):
    def __init__(self, args):
        super(EncoderIT, self).__init__()
        "初始化参数两次，两套参数。"
        # 初始化EncoderI
        self.EncoderI = supBackOne(args)
        # 初始化EncoderT
        self.EncoderT = supBackTwo(args)

    def forward(self, x):
        featuresI, sort_Loss_I = self.EncoderI(x)
        featuresT, sort_Loss_T = self.EncoderT(x)
        # 返回variance and in-variance features
        sort_Loss = sort_Loss_I+sort_Loss_T
        return featuresI, featuresT, sort_Loss
