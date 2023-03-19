from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("/task/yaml/ctw.yaml", "basic")
DICT = {
    "500": base.DanConfig(pathSet.modelPath[0]),
    "1000": base.DanConfig(pathSet.modelPath[0]),
    "1500": base.DanConfig(pathSet.modelPath[0]),
    "2000": base.DanConfig(pathSet.modelPath[0])
}

if __name__ == '__main__':
    for k in DICT:
        cfgs = DICT[k](pathSet.modelRoot)
        runner = baseline.BaselineDAN(cfgs)
        runner.run(pathSet.modelRoot, False)
        print(k, "Done")
