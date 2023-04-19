from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/mjst.yaml", "basic")
DICT = {
    "task0": base.ColoredDanConfig,
    "task1": base.ColoredDanConfig,
    "task2": base.ColoredDanConfig,
    "task3": base.ColoredDanConfig
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        cfgs = DICT[k](
            pathSet,
            i,
            [pathSet.multiTrain['nips14'], pathSet.multiTrain['cvpr16']],
            [pathSet.multiTrain['iiit5k']],
            mode="Train"
        )
        runner = baseline.BaselineDAN(cfgs)
        runner.run(pathSet.modelRoot, False)
        print(k, "Done")
