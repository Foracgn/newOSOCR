def getTrainCfg():
    return {
        'state': 'Train',
        'epoch': 10,
        'showInterval': 50,
        'testInterval': 1000,
        'testMiter': 100,
        'printNet': False,
    }


def getTestCfg():
    return {
        'state': 'Test',
        'epoch': 10,
        'showInterval': 50,
        'testInterval': 1000,
        'testMiter': 100000,
        'printNet': False,
    }


def getSaveCfg(saveRoot):
    return{
        'savingIterInterval': 20000,
        'savingEpochInterval': 1,
        'savingPath': saveRoot
    }
