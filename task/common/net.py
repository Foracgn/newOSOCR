import torch


def getNetConfig():
    pass


def loadNet(cfgs):
    # 特征提取器 feature extra
    modelFE = cfgs.netConfigs['FE'](**cfgs.netConfigs['FEConfigs'])
    cfgs.netConfigs['CAMConfigs']['scales'] = modelFE.getShape()
    if cfgs.netConfigs['initStateDictFE'] is not None:
        modelFE.loadStateDict(torch.load(cfgs.netConfigs['initStateDictFE']))
    modelFE.cuda()
    # 卷积对齐模块
    modelCAM = cfgs.netConfigs['CAM'](**cfgs.netConfigs['CAMConfigs'])
    if cfgs.netConfigs['initStateDictCAM'] is not None:
        modelCAM.loadStateDict(torch.load(cfgs.netConfigs['initStateDictCAM']))
    modelCAM.cuda()
    # 去耦解码器
    modelDTD = cfgs.netConfigs['DTD'](**cfgs.netConfigs['DTDConfigs'])
    if cfgs.netConfigs['initStateDictDTD'] is not None:
        modelDTD.loadStateDict(torch.load(cfgs.netConfigs['initStateDictDTD']))
    modelDTD.cuda()
    # 位置编码 Positional Encoding
    modelPE = cfgs.netConfigs['PE'](**cfgs.netConfigs['PEConfigs'])
    if cfgs.netConfigs['initStateDictPE'] is not None:
        modelPE.loadStateDict(torch.load(cfgs.netConfigs['initStateDictPE']))
    modelPE.cuda()

    return modelFE, modelCAM, modelDTD, modelPE


def TrainOrEval(model, state='Train'):
    for one in model:
        if state == 'Train':
            one.Train()
        else:
            one.Eval()


def ZeroGrad(model):
    for one in model:
        one.zero_grad()


def FlattenLabel(target):
    labelFlatten = []
    labelLength = []
    for i in range(0, target.size()[0]):
        curLabel = target[i].tolist()
        labelFlatten += curLabel[:curLabel.index(0)+1]
        labelLength.append(curLabel.index(0)+1)
    labelFlatten = torch.LongTensor(labelFlatten)
    labelLength = torch.IntTensor(labelLength)
    return labelFlatten, labelLength
