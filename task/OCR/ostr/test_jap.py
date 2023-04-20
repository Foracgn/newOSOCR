from __future__ import print_function
from task.model.DAN import base
from task.model import path
from task.framework import baseline

DICT = {
    "dabjpmltch_osr": base.FreeDictDanConfig,
    "dabjpmltch_sharedkanji": base.FreeDictDanConfig,
    "dabjpmltch_nohirakata": base.FreeDictDanConfig,
    "dabjpmltch_kanji": base.FreeDictDanConfig
}

if __name__ == '__main__':
    for i, k in enumerate(DICT):
        pathSet = path.Path("./task/yaml/ostr.yaml", "basic", "rej", "dict"+str(i))
        cfgs = DICT[k](
            pathSet.modelPath[0],
            pathSet.testDict,
            [pathSet.multiTest[k]],
            [pathSet.multiTest[k]],
            mode="Test"
        )
        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(pathSet.modelRoot, False)
        print(k, "Done")
