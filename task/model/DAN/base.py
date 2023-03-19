from task.common import train, net, dataset, optimizer, loss

root = [""]
T = 2
saveRoot = ""


class DanConfig:
    def __init__(self, modelPath):
        self.datasetConfigs = dataset.getCompareDatasetConfig("", "", "", "")
        self.globalConfigs = train.getTrainCfg()
        self.netConfigs = net.getNetConfig(modelPath, T)
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(saveRoot)
        self.lossWeight = loss.cls_emb[1]
