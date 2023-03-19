from __future__ import print_function
from task.model.DAN import base
import yaml
from task.framework import baseline

filepath = "/task/yaml/DAN.yaml"
ctwConfig = yaml.load(filepath)
DICT = {
    "500": base.DanConfig(),
    "1000": base.DanConfig(),
    "1500": base.DanConfig(),
    "2000": base.DanConfig()
}
root = "/"

if __name__ == '__main__':
    for k in DICT:
        cfgs = DICT[k](root)
        runner = baseline.BaselineDAN(cfgs)

        print(k, "Done")
