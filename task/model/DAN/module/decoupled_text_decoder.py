from torch import nn
import torch
from neko_sdk.ocr_modules.neko_score_merging import scatter_cvt


# TODO 对比现有DAN网络的DTD模块
class DecoupledTextDecoder(nn.Module):

    def __init__(self, numChannel, dropout=0.3, xTrapParam=None):
        # 参数
        self.context_free_pred = None
        self.ALPHA = None
        self.STA = None
        self.UNK = None
        self.UNK_SCR = None

        # 初始化
        super(DecoupledTextDecoder, self).__init__()
        self.numChannel = numChannel
        self.dropout = dropout
        self.xTrapParam = xTrapParam
        self.setModule()
        self.baseline = 0

    def setModule(self, dropout=0.3):
        self.STA = torch.nn.Parameter(self.normedInit(self.numChannel))
        self.UNK = torch.nn.Parameter(self.normedInit(self.numChannel))
        self.UNK_SCR = torch.nn.Parameter(torch.zeros([1, 1]), requires_grad=True)
        self.ALPHA = torch.nn.Parameter(torch.ones([1, 1]), requires_grad=True)
        self.context_free_pred = torch.nn.Linear(self.numChannel, self.numChannel)

        self.register_parameter("STA", self.STA)
        self.register_parameter("UNK", self.UNK)
        self.register_parameter("UNK_SCR", self.UNK_SCR)
        self.register_parameter("ALPHA", self.ALPHA)

    def forward(self, feature, protos, labels, A, hype, textLength, test=False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        A, C = self.sample(feature, A, nB, nC, nH, nW, nT)
        C = nn.functional.dropout(C, p=0.3, training=self.training)
        if not test:
            return self.forwardTrain(protos, labels, nB, C, nT, textLength, A, nW, nH)
        else:
            out, outL = self.forwardTest(protos, labels, nB, C, nT)
            return out, outL, A

    def forwardTrain(self, protos, labels, nB, C, nT, textLength, A, nW, nH, hype=None):
        steps = int(textLength.max())
        outRes, _ = self.loop(C, protos, steps, nB, hype)
        outRes = self.predict(outRes, labels, textLength, nB, nT)
        # _ = self.predict(_, labels, textLength, nB, nT)
        return outRes, _

    def forwardTest(self, protos, labels, nB, C, nT):
        outRes, _ = self.loop(C, protos, nT, nB, None)
        outLength = self.prob_length(outRes, nT, nB)
        output = self.predict(outRes, labels, outLength, nB, nT)
        return output, outLength

    def loop(self, C, protos, steps, nB, hype):
        outRes = torch.zeros(steps, nB, protos.shape[0] + 1).type_as(C.data) + self.UNK_SCR
        attentionMap = torch.zeros(steps, nB, protos.shape[0] + 1).type_as(C.data) + self.UNK_SCR

        # context_free_predict:torch.nn.Linear(self.numChannel, self.numChannel)
        hidden = self.context_free_pred(C)
        cfPredict = hidden.matmul(protos.t())

        # 2范数
        cfCos = cfPredict / (hidden.norm(dim=-1, keepdim=True) + 0.0009)

        outRes[:steps, :, :] = torch.cat(
            [
                cfPredict[:steps, :, :] * self.ALPHA,
                self.UNK_SCR.repeat(steps, nB, 1)
            ],
            dim=-1
        )
        attentionMap[:steps, :, :-1] = cfCos[:steps, :, :]

        return outRes, attentionMap

    @staticmethod
    def normedInit(numChannel):
        data = torch.rand(1, numChannel)
        return data / torch.norm(data, dim=-1, keepdim=True)

    @staticmethod
    def sample(feature, A, nB, nC, nH, nW, nT):
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        # weighted sum
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)
        return A, C

    @staticmethod
    def predict(res, label, outLength, nB, nT):
        scores = scatter_cvt(res, label)
        start = 0
        output = torch.zeros(int(outLength.sum()), label.max().item() + 1).type_as(res.data)
        for i in range(0, nB):
            curLength = int(outLength[i])
            usedLength = min(nT, curLength)
            output[start: start + usedLength] = scores[0:usedLength, i, :]
            start += usedLength
        return output

    @staticmethod
    def prob_length(out, steps, nB):
        outLength = torch.zeros(nB)
        for i in range(0, steps):
            tens = out[i, :, :]
            res = tens.topk(1)[1].squeeze(-1)
            for j in range(nB):
                if outLength[j].item() == 0 and res[j] == 0:
                    outLength[j] = i + 1
        for j in range(nB):
            if outLength[j] == 0:
                outLength[j] = steps + 1
        return outLength
