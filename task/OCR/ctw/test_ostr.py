from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/ctw.yaml", "basic")
DICT = {
    "task0": base.DanConfig,
    "task1": base.DanConfig,
    "task2": base.DanConfig
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        cfgs = DICT[k](pathSet, i, "Test")
        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(pathSet.modelRoot, reject=True)
        print(k, "Done")
