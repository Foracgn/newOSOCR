from torch import nn
import torch
from neko_sdk.ocr_modules.neko_score_merging import scatter_cvt


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

    def sample(self, feature, A):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        # weighted sum
        C = self.getC(feature, A, nB, nC, nH, nW, nT)
        return A, C

    def forward(self, feature, protos, semblance, labels, A, hype, textLength, test=False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        A, C = self.sample(feature, A)
        C = nn.functional.dropout(C, p=0.3, training=self.training)
        if not test:
            return self.forwardTrain(protos, semblance, labels, nB, C, nT, textLength, A, nW, nH)
        else:
            out, outL = self.forwardTest(protos, semblance, labels, nB, C, nT)
            return out, outL, A

    def forwardTrain(self, protos, semblance, labels, nB, C, nT, textLength, A, nW, nH, hype=None):
        steps = int(textLength.max())
        outCls, outCos = self.loop(C, protos, semblance, labels, steps, nB, hype)
        outCls = self.predict(outCls, labels, textLength, nB, nT)
        outCos = self.predict(outCos, labels, textLength, nB, nT)
        return outCls, outCos

    def forwardTest(self, protos, semblance, labels, nB, C, nT):
        outCls, _ = self.loop(C, protos, semblance, labels, nT, nB, None)
        outLength = self.prob_length(outCls, nT, nB)
        output = self.pred(outCls, labels, outLength, nB, nT)
        return output, outLength

    def loop(self, C, protos, semblance, labels, steps, nB, hype):
        out_res_cf = torch.zeros(steps, nB, protos.shape[0] + 1).type_as(C.data) + self.UNK_SCR
        sim_score = torch.zeros(steps, nB, protos.shape[0] + 1).type_as(C.data) + self.UNK_SCR
        # hidden=C;
        hidden = self.context_free_pred(C)
        cfPredict = hidden.matmul(protos.t())

        cfCos = cfPredict / (hidden.norm(dim=-1, keepdim=True) + 0.0009)

        out_res_cf[:steps, :, :] = torch.cat(
            [
                cfPredict[:steps, :, :] * self.ALPHA,
                self.UNK_SCR.repeat(steps, nB, 1)
            ],
            dim=-1
        )
        sim_score[:steps, :, :-1] = cfCos[:steps, :, :]

        return out_res_cf, sim_score

    @staticmethod
    def normedInit(numChannel):
        data = torch.rand(1, numChannel)
        return data / torch.norm(data, dim=-1, keepdim=True)

    @staticmethod
    def getC(feature, A, nB, nC, nH, nW, nT):
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)
        return C

    @staticmethod
    def predict(res, label, outLength, nB, nT):
        scores = scatter_cvt(res, label)
        start = 0
        output = torch.zeros(int(outLength.sum()), label.max().item + 1).type_as(res.data)
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
