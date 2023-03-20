import task.model.dataset
from torchvision import transforms
from task.model.dataset import lmdbDataset


def getDatasetConfig(maxT, root, dictPath):
    return {
        'dataset': lmdbDataset,
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


# Todo:dataset train
def getCompareDatasetConfig(trainRoot, trainDict, testRoot, testDict, maxT=25):
    return {
        'datasetTrain': lmdbDataset,
        'datasetTrainConfigs': {
            'repeat': 1,
            'roots': [trainRoot],
            'imgH': 32,
            'imgW': 64,
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
        'datasetTest': lmdbDataset,
        'datasetTestConfigs': {
            'roots': [testRoot],
            'imgH': 32,
            'imgW': 64,
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
