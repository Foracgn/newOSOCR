from torch import nn
import torch


class DecoupledTextDecoder(nn.Module):

    def __init__(self, nChannels, dropout=0.3, xTrapParam=None):
        super(DecoupledTextDecoder, self).__init__()
        self.nChannels = nChannels
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
        self.STA = torch.nn.Parameter(self.normedInit(self.nChannel))
        self.UNK = torch.nn.Parameter(self.normedInit(self.nChannel))
        self.UNK_SCR = torch.nn.Parameter(torch.zeros([1, 1]), requires_grad=True)
        self.ALPHA = torch.nn.Parameter(torch.ones([1, 1]), requires_grad=True)
        self.contextFreePredict = torch.nn.Linear(self.nChannel, self.nChannel)

        self.registerParameter("STA", self.STA)
        self.registerParameter("STA", self.UNK)
        self.registerParameter("STA", self.UNK_SCR)
        self.registerParameter("STA", self.ALPHA)

    @staticmethod
    def normedInit(nChannel):
        data = torch.rand(1, nChannel)
        return data / torch.norm(data, dim=-1, keepdim=True)