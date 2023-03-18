import task.model.dataset
from torchvision import transforms


def getDatasetConfig(maxT, root, dictPath):
    return {
        'dataset': 1,
        'datesetConfigs': {
            'root': root,
            'imgHeight': 32,
            'imgWidth': 128,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'globalState': 'Test',
            "maxT": maxT,
        },
        'dataloaderTest': {
            'batch_size': 128,
            'shuffle': False,
            'num_workers': 4,
        },
        'caseSensitive': False,
        'dictPath': dictPath
    }


# Todo:test dataloader
def getCompareDatasetConfig(trainRoot, trainDict, testRoot, testDict, maxT=25):
    return {
        'datasetTrain': 1,
        'datasetTrainConfigs': {
            'repeat': 1,
            'root': trainRoot,
            'imgHeight': 32,
            'imgWidth': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'globalState': 'Train',
            'maxT': maxT,
        },
        'dataloaderTrain': {
            'batch_size': 160,
            'shuffle': True,
            'num_workers': 5,
        },
        'trainCaseSensitive': False,
        'trainMeta': trainDict,
        'datasetTest': 1,
        'datasetTestConfigs': {
            'root': testRoot,
            'imgHeight': 32,
            'imgWidth': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'globalState': 'Test',
            'maxT': maxT,
        },
        'dataloaderTest': {
            'batch_size': 8,
            'shuffle': False,
            'num_workers': 5,
        },
        'testCaseSensitive': False,
        'testMeta': testDict,
        'caseSensitive': False,
    }
