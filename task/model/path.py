import yaml


class Path:
    def __init__(self, constPath, setType, ):
        self.modelPath = []
        data = open(constPath, 'r').read()
        setConfigs = yaml.load(data, Loader=yaml.FullLoader)
        self.modelRoot = setConfigs['modelRoot']
        self.modelPath[0] = self.modelRoot + setConfigs[setType]['FE']
        self.modelPath[1] = self.modelRoot + setConfigs[setType]['CAM']
        self.modelPath[2] = self.modelRoot + setConfigs[setType]['DTD']
        self.modelPath[3] = self.modelRoot + setConfigs[setType]['PE']
