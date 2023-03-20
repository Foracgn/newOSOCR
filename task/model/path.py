import yaml


class Path:
    def __init__(self, constPath, setType, ):
        # 参数
        self.modelPath = []
        self.trainRoot = []
        self.trainDict = []
        # 初始化
        data = open(constPath, 'r').read()
        setConfigs = yaml.load(data, Loader=yaml.FullLoader)
        self.modelRoot = setConfigs['modelRoot']
        for i in range(0, 4):
            taskPath = [
                self.modelRoot + setConfigs[setType]['FE' + str(i)],
                self.modelRoot + setConfigs[setType]['CAM' + str(i)],
                self.modelRoot + setConfigs[setType]['DTD'+str(i)],
                self.modelRoot + setConfigs[setType]['PE'+str(i)]
            ]
            self.modelPath.append(taskPath)

        self.datasetRoot = setConfigs['datasetRoot']

        self.testRoot = self.datasetRoot + setConfigs['test']['root']
        self.testDict = self.datasetRoot + setConfigs['test']['dict']

        for i in range(0, 4):
            self.trainRoot.append(self.datasetRoot + setConfigs['train']['root'+str(i)])
            self.trainDict.append(self.datasetRoot + setConfigs['train']['dict'+str(i)])
