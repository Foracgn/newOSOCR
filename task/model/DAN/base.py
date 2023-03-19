from task.common import train, net, dataset, optimizer, loss

metaPath = ""
root = [""]
T = 2
saveRoot = ""


class DanConfig:
    def __init__(self):
        self.globalConfigs = train.getTrainCfg()
        self.netConfigs = net.getNetConfig(metaPath, "E9", T, root)
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg(saveRoot)
        self.datasetConfigs = dataset.getCompareDatasetConfig("", "", "", "")
        self.lossWeight = loss.cls_emb[1]
