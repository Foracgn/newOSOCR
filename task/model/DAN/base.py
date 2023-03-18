from task.common import train, net, dataset, optimizer, loss


class DanConfig:
    def __init__(self):
        self.globalConfigs = train.getTrainCfg()
        self.netConfigs = net.getNetConfig()
        self.optimizerConfigs = optimizer.getOpt()
        self.savingConfigs = train.getSaveCfg()
        self.datasetConfigs = dataset.getCompareDatasetConfig("", "", "", "")
        self.lossWeight = loss.cls_emb[1]
