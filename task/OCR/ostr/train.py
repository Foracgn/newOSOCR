from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/ostr.yaml", "basic")

if __name__ == '__main__':
    trainSet = []
    for _, dataset in enumerate(pathSet.multiTrain):
        trainSet.append(dataset)
    cfgs = base.FreeDictDanConfig(
        pathSet.modelPath[0],
        pathSet.trainDict[0],
        trainSet,
        trainSet,
        mode="Train"
    )
    runner = baseline.BaselineDAN(cfgs)
    runner.run(pathSet.modelRoot, False)
    print("train task ostr Done")
