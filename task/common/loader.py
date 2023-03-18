def loadDataset(cfgs, DataLoader):
    trainDataSet = cfgs.datasetConfigs['datasetTrain'](**cfgs.datasetConfigs['datasetTrainConfigs'])
    trainLoader = DataLoader(trainDataSet, **cfgs.datasetConfigs['dataloaderTrain'])

    testDataSet = cfgs.datasetConfigs['datasetTest'](**cfgs.datasetConfigs['datasetTestConfigs'])
    testLoader = DataLoader(testDataSet, **cfgs.datasetConfigs['dataloaderTest'])

    return trainLoader, testLoader
