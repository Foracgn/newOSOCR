from task.common import train, net, dataset, optimizer, loss

root = [""]
T = 2
saveRoot = ""


class DanConfig:
    def __init__(self, pathSet, num):
        self.datasetConfigs = dataset.getCompareDatasetConfig(
            pathSet.trainRoot[num],
            pathSet.trainDict[num],
            pathSet.testRoot[num],
            pathSet.testRoot[num],
            T
        )
        self.globalConfigs = train.getTrainCfg()
        self.netConfigs = net.getNetConfig(pathSet.trainDict[num], pathSet.modelPath[num], T)
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(pathSet.modelPath[num])
        self.lossWeight = loss.cls_emb[1]
