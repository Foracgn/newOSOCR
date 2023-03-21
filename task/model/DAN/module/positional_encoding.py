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
        return self.dwcore.sample_charset_by_text(label)

    def encode(self, proto, pLabel, tdict, batch):
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
