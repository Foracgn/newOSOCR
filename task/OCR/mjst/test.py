from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/mjst.yaml", "basic", "test")
DICT = {
    "CUTE": base.FreeDictDanConfig,
    "SVT": base.FreeDictDanConfig,
    "IC03_867": base.FreeDictDanConfig,
    "IC13_1015": base.FreeDictDanConfig,
    "iiit5k": base.FreeDictDanConfig
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        cfgs = DICT[k](
            pathSet.modelPath[0],
            pathSet.trainDict[0],
            [pathSet.multiTest[k]],
            [pathSet.multiTest[k]],
            mode="Test"
        )
        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(pathSet.modelRoot, False)
        print(k, "Done")
