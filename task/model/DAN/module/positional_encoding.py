from torch import nn
import torch
from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core import neko_nonsematical_prototype_core_basic
import regex


class PositionalEncoding(nn.Module):
    def __init__(self, metaPath, numChannel, caseSensitive, backbone=None, valFrac=0.8):
        # 参数
        self.dwcore = None

        # 初始化
        super(PositionalEncoding, self).__init__()
        self.EOS = 0
        self.numChannel = numChannel
        self.caseSensitive = caseSensitive
        self.metaPath = metaPath
        self.setCore(backbone, valFrac)

    def setCore(self, backbone=None, valFrac=0.8):
        meta = torch.load(self.metaPath)
        self.dwcore = neko_nonsematical_prototype_core_basic(
            self.numChannel,
            meta,
            backbone,
            # none backbone
            None,
            {
                "master_share": not self.caseSensitive,
                "max_batch_size": 512,
                "val_frac": valFrac,
                "neg_servant": True
            },
            dropout=0.3
        )

    def sampleTrain(self, label):
        # entrance:make proto
        return self.dwcore.sample_charset_by_text(label)

    def encode(self, tdict, batch):
        if not self.caseSensitive:
            batch = [ch.lower() for ch in batch]
        return self.encodeNaive(tdict, batch)

    def encodeNaive(self, tdict, batch):
        maxLen = max([len(regex.findall(r'\X', s, regex.U)) for s in batch])
        out = torch.zeros(len(batch), maxLen + 1).long() + self.EOS
        for i in range(0, len(batch)):
            curEncoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                       for char in regex.findall(r'\X', batch[i], regex.U)])
            out[i][0:len(curEncoded)] = curEncoded
        return out

    def dumpAll(self):
        return self.dwcore.dump_all()

    @staticmethod
    def decode(netOUT, length, tdict, thresh=None):
        outDecode = []
        outProbability = []
        netOUT = nn.functional.softmax(netOUT, dim=1)
        # TODO netOUT的内容
        for i in range(0, length.shape[0]):
            curIndexList = netOUT[
                           int(length[:i].sum()): int(length[:i].sum() + length[i])
                           ].topk(1)[1][:, 0].tolist()

            curText = ''.join([tdict[_] if 0 < _ <= len(tdict) else '' for _ in curIndexList])
            curProbability = netOUT[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
            curProbability = torch.exp(torch.log(curProbability).sum() / curProbability.size()[0])
            if thresh is not None:
                filteredText = []
                for j in range(len(curText)):
                    if curProbability[j] > thresh:
                        filteredText.append(curText[j])
                    else:
                        filteredText.append('⑨')
                curText = ''.join(filteredText)
            outDecode.append(curText)
            outProbability.append(curProbability)
        return outDecode, outProbability


class PositionalEncodingOracle(PositionalEncoding):

    def encodeNaive(self, tdict, batch):
        maxLen = max([len(regex.findall(r'\X', s, regex.U)) for s in batch])
        out = torch.zeros(len(batch), maxLen + 1).long() + self.EOS
        for i in range(0, len(batch)):
            curEncoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                       for char in regex.findall(r'\X', batch[i], regex.U)])
            out[i][0:len(curEncoded)] = curEncoded
        return out

    @staticmethod
    def decode(netOUT, length, tdict, thresh=None):
        outDecode = []
        outProbability = []
        netOUT = nn.functional.softmax(netOUT, dim=1)
        # TODO netOUT的内容
        for i in range(0, length.shape[0]):
            curIndexList = netOUT[
                           int(length[:i].sum()): int(length[:i].sum() + length[i])
                           ].topk(1)[1][:, 0].tolist()

            curText = ''.join([tdict[_] if 0 < _ <= len(tdict) else '' for _ in curIndexList])
            curProbability = netOUT[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
            curProbability = torch.exp(torch.log(curProbability).sum() / curProbability.size()[0])
            if thresh is not None:
                filteredText = []
                for j in range(len(curText)):
                    if curProbability[j] > thresh:
                        filteredText.append(curText[j])
                    else:
                        filteredText.append('⑨')
                curText = ''.join(filteredText)
            outDecode.append(curText)
            outProbability.append(curProbability)
        return outDecode, outProbability
