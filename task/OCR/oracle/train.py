from __future__ import print_function
from task.model.DAN import base
from task.framework import baseline
import yaml

pathSet = path.Path("./task/yaml/ostr.yaml", "basic")

if __name__ == '__main__':
    cfgs = base.OracleDanConfig(
        pathSet.modelPath[0],
        pathSet.trainDict[0],
        trainSet,
        trainSet,
        mode="Train"
    )
    runner = baseline.BaselineDAN(cfgs)
    runner.run(pathSet.modelRoot, False)
    print("train task ostr Done")
