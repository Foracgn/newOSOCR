from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

DICT = {
    "dict50": "dict50",
    "dict100": "dict100",
    "dict150": "dict150",
    "dict200": "dict200",
    "dict250": "dict250",
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        pathSet = path.Path("./task/yaml/hwdb.yaml", "basic", "rej", k)
        cfgs = base.DanConfig(pathSet, i, "Test")
        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(pathSet.modelRoot, reject=True)
        print(k, "Done")
