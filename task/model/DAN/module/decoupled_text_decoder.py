from torch import nn
import torch


class DecoupledTextDecoder(nn.Module):

    def __init__(self, numChannel, dropout=0.3, xTrapParam=None):
        super(DecoupledTextDecoder, self).__init__()
        self.numChannel = numChannel
        self.dropout = dropout
        self.xTrapParam = xTrapParam
        self.setModule()
        self.baseline = 0

        # 参数
        self.contextFreePredict = None
        self.ALPHA = None
        self.STA = None
        self.UNK = None
        self.UNK_SCR = None

    def setModule(self, dropout=0.3):
        self.STA = torch.nn.Parameter(self.normedInit(self.numChannel))
        self.UNK = torch.nn.Parameter(self.normedInit(self.numChannel))
        self.UNK_SCR = torch.nn.Parameter(torch.zeros([1, 1]), requires_grad=True)
        self.ALPHA = torch.nn.Parameter(torch.ones([1, 1]), requires_grad=True)
        self.contextFreePredict = torch.nn.Linear(self.numChannel, self.numChannel)

        self.register_parameter("STA", self.STA)
        self.register_parameter("UNK", self.UNK)
        self.register_parameter("UNK_SCR", self.UNK_SCR)
        self.register_parameter("ALPHA", self.ALPHA)

    @staticmethod
    def normedInit(numChannel):
        data = torch.rand(1, numChannel)
        return data / torch.norm(data, dim=-1, keepdim=True)
