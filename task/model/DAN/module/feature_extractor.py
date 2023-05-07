from torch import nn
import torch
import neko_sdk.encoders.ocr_networks.dan.dan_reslens_naive as rescco


class FeatureExtractor(nn.Module):
    def __init__(self, strides, compressLayer, shape, hardness=2, oupch=512, expf=1):
        super(FeatureExtractor, self).__init__()
        self.model = rescco.res_naive_lens45(strides, compressLayer, hardness, oupch=oupch, inpch=shape[0], expf=expf)
        self.shape = shape

    def forward(self, data):
        feature, grid = self.model(data)
        return feature, grid

    def getShape(self):
        data = torch.rand(1, self.shape[0], self.shape[1], self.shape[2])
        feature, grid = self.model(data)
        return [one.size()[1:] for one in feature]
