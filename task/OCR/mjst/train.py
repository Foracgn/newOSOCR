from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/mjst.yaml", "basic")

if __name__ == '__main__':
    cfgs = base.FreeDictDanConfig(
        pathSet.modelPath[0],
        pathSet.trainDict[0],
        [pathSet.multiTrain['nips14'], pathSet.multiTrain['cvpr16']],
        [pathSet.multiTrain['iiit5k']],
        mode="Train"
    )
    runner = baseline.BaselineDAN(cfgs)
    runner.run(pathSet.modelRoot, False)
    print("task mjst Done")
