import torch
from task.common.module import convolutional_alignment, decoupled_text_decoder, feature_extractor, positional_encoding
from task.model.DAN.module import convolutional_alignment as cam
from task.model.DAN.module import decoupled_text_decoder as dtd
from task.model.DAN.module import feature_extractor as fe
from task.model.DAN.module import positional_encoding as pe


def getNetConfig(metaPath, modelPath, maxT, valFrac=0.8):
    configs = makeNetConfig(
        fe.FeatureExtractor,
        cam.ConvolutionAlignment,
        dtd.DecoupledTextDecoder,
        pe.PositionalEncoding,
        0.5,
        metaPath,
        maxT,
        valFrac
    )
    makeToken(configs, modelPath)


def makeNetConfig(FE, CAM, DTD, PE, hardness, metaPath, maxT, valFrac=0.8):
    return {
        'FE': FE,
        'FE_args': feature_extractor.getFEConfig(hardness),
        'CAM': CAM,
        'CAM_args': convolutional_alignment.getCAMConfig(maxT),
        'DTD': DTD,
        'DTD_args': decoupled_text_decoder.getDTDConfig(),
        'PE': PE,
        'PE_args': positional_encoding.getPEConfig(metaPath, valFrac)
    }


def makeToken(configs, modelPath):
    configs['initStateDictFE'] = modelPath[0]
    configs['initStateDictCAM'] = modelPath[1]
    configs['initStateDictDTD'] = modelPath[2]
    configs['initStateDictPE'] = modelPath[3]

    return configs


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
        labelFlatten += curLabel[:curLabel.index(0) + 1]
        labelLength.append(curLabel.index(0) + 1)
    labelFlatten = torch.LongTensor(labelFlatten)
    labelLength = torch.IntTensor(labelLength)
    return labelFlatten, labelLength
