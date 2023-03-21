from torch import nn
import torch
from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core import neko_nonsematical_prototype_core_basic


class PositionalEncoding(nn.Module):
    def __init__(self, metaPath, numChannel, caseSensitive, backbone=None, valFrac=0.8):
        super(PositionalEncoding, self).__init__()
        self.EOS = 0
        self.numChannel = numChannel
        self.caseSensitive = caseSensitive
        self.metaPath = metaPath
        self.setCore(backbone, valFrac)

        # 参数
        self.DWCore = None

    def setCore(self, backbone=None, valFrac=0.8):
        meta = torch.load(self.metaPath)
        self.DWCore = neko_nonsematical_prototype_core_basic(
            self.numChannel,
            meta,
            backbone,
            None,
            {
                "masterShare": not self.caseSensitive,
                "max_batch_size": 512,
                "val_frac": valFrac,
                "neg_servant": True
            },
            dropout=0.3
        )
