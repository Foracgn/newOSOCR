from __future__ import print_function
from task.model.DAN import base
import yaml
from task.framework import baseline
import os

filepath = "/task/yaml/ctw.yaml"
ctwConfig = yaml.load(filepath)
DICT = {
    "500": base.DanConfig(),
    "1000": base.DanConfig(),
    "1500": base.DanConfig(),
    "2000": base.DanConfig()
}
modelRoot = os.path.realpath(__file__)+"/"

if __name__ == '__main__':
    for k in DICT:
        cfgs = DICT[k](modelRoot)
        runner = baseline.BaselineDAN(cfgs)
        runner.run(modelRoot, False)
        print(k, "Done")
