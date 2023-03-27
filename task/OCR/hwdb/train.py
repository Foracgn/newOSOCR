from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/hwdb.yaml", "basic")
DICT = {
    "task0": base.DanConfig,
    "task1": base.DanConfig,
    "task2": base.DanConfig,
    "task3": base.DanConfig
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        cfgs = DICT[k](pathSet, i, mode="Train")
        runner = baseline.BaselineDAN(cfgs)
        runner.run(pathSet.modelRoot, False)
        print(k, "Done")
