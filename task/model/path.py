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
            self.modelPath[i][0] = self.modelRoot + setConfigs[setType]['FE'+str(i)]
            self.modelPath[i][1] = self.modelRoot + setConfigs[setType]['CAM'+str(i)]
            self.modelPath[i][2] = self.modelRoot + setConfigs[setType]['DTD'+str(i)]
            self.modelPath[i][3] = self.modelRoot + setConfigs[setType]['PE'+str(i)]

        self.datasetRoot = setConfigs['datasetRoot']

        self.testRoot = self.datasetRoot + setConfigs['test']['root']
        self.testDict = self.datasetRoot + setConfigs['test']['dict']

        for i in range(0, 4):
            self.trainRoot[i] = self.datasetRoot + setConfigs['train']['root'+str(i)]
            self.trainDict[i] = self.datasetRoot + setConfigs['train']['dict'+str(i)]
