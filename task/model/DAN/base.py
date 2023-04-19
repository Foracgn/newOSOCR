from task.common import train, net, dataset, optimizer, loss


class DanConfig:
    def __init__(self, pathSet, num, mode="Test"):
        self.datasetConfigs = dataset.getCompareDatasetConfig(
            pathSet.trainRoot[num],
            pathSet.trainDict[num],
            pathSet.testRoot,
            pathSet.testDict,
            maxT=2
        )
        if mode == "Train":
            self.globalConfigs = train.getTrainCfg()
        else:
            self.globalConfigs = train.getTestCfg()
        self.netConfigs = net.getNetConfig(pathSet.trainDict[num], pathSet.modelPath[num], maxT=2, mode=mode)
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(pathSet.modelPath[num])
        self.lossWeight = loss.cls_emb[1]


class ColoredDanConfig:
    def __init__(self, pathSet, num, trainRoot, testRoot, mode="Train"):
        self.datasetConfigs = dataset.getColoredDataset(trainRoot, testRoot)

        if mode == "Train":
            self.globalConfigs = train.getTrainCfg()
        else:
            self.globalConfigs = train.getTestCfg()

        self.netConfigs = net.getNetConfig(pathSet.trainDict[0], pathSet.modelPath[num], maxT=25, mode=mode)
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(pathSet.modelPath[num])
        self.lossWeight = loss.cls_emb[1]
