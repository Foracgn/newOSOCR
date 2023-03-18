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


def getSaveCfg():
    return 0
