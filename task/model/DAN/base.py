from task.common import train, net, dataset, optimizer, loss

root = [""]
T = 2
saveRoot = ""


class DanConfig:
    def __init__(self, pathSet, num, mode="Test"):
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
        self.netConfigs = net.getNetConfig(pathSet.trainDict[num], pathSet.modelPath[num], T, mode=mode)
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(pathSet.modelPath[num])
        self.lossWeight = loss.cls_emb[1]


class ColoredDanConfig:
    def __init__(self, pathSet, num, mode="Train"):
        self.datasetConfigs = dataset.getColoredDataset(
            [pathSet.multiTrain['nips14'], pathSet.multiTrain['cvpr16']],
            [pathSet.multiTrain['iiit5k']]
        )

        if mode == "Train":
            self.globalConfigs = train.getTrainCfg()
        else:
            self.globalConfigs = train.getTestCfg()

        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(pathSet.modelPath[num])
        self.lossWeight = loss.cls_emb[1]
