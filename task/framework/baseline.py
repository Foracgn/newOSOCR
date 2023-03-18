from torch.utils.data import DataLoader
from task.common import loader
from task.model import loss


class BaselineDAN:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.setupDataloader()
        self.setup()

        # 类内变量
        self.trainLoader = None
        self.testLoader = None
        self.allLoader = None
        self.loss = None
        self.testReject = None
        self.testAccuracy = None
        self.trainAccuracy = None

    def setup(self):
        pass

    def setupDataloader(self):
        if "datasetTrain" in self.cfgs.datasetConfigs:
            self.trainLoader, self.testLoader = loader.loadDataset(self.cfgs, DataLoader)
            self.setCounters()

    def setCounters(self):
        self.trainAccuracy = self.getAccuracy('train accuracy:', self.cfgs.datasetConfigs['trainCaseSensitive'])
        self.testAccuracy = self.getAccuracy('test accuracy:', self.cfgs.datasetConfigs['testCaseSensitive'])
        self.testReject = self.getReject('test reject:', self.cfgs.datasetConfigs['testCaseSensitive'])
        self.loss = self.getLoss(self.cfgs.globalConfig['showInterval'])

    @staticmethod
    def getAccuracy(key, sensitive):
        return loss.AccuracyCounter(key, sensitive)

    @staticmethod
    def getReject(key, sensitive):
        return loss.RejectAccuracyCounter(key, sensitive)

    @staticmethod
    def getLoss(interval):
        return loss.LossCounter(interval)
