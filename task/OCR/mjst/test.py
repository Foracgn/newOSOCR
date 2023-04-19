from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

pathSet = path.Path("./task/yaml/mjst.yaml", "basic", "test")
DICT = {
    "CUTE": base.ColoredDanConfig,
    "SVT": base.ColoredDanConfig,
    "IC03_867": base.ColoredDanConfig,
    "IC13_1015": base.ColoredDanConfig
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        cfgs = DICT[k](
            pathSet,
            i,
            [],
            [pathSet.multiTest[k]],
            mode="Test"
        )
        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(pathSet.modelRoot, False)
        print(k, "Done")
