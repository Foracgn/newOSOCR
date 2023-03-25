from task.common import train, net, dataset, optimizer, loss

root = [""]
T = 2
saveRoot = ""


class DanConfig:
    def __init__(self, pathSet, num, mode="Train"):
        self.datasetConfigs = dataset.getCompareDatasetConfig(
            pathSet.trainRoot[num],
            pathSet.trainDict[num],
            pathSet.testRoot,
            pathSet.testDict,
            T
        )
        if mode == "Train":
            self.globalConfigs = train.getTrainCfg()
        else:
            self.globalConfigs = train.getTestCfg()
        self.netConfigs = net.getNetConfig(pathSet.trainDict[num], pathSet.modelPath[num], T, "Train")
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(pathSet.modelPath[num])
        self.lossWeight = loss.cls_emb[1]
