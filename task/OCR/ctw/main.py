from __future__ import print_function
from task.model.ctw import config
import yaml
from task.framework import baseline

filepath = "/task/yaml/ctw.yaml"
ctwConfig = yaml.load(filepath)
DICT = {
    "500": config.DanConfig(),
    "1000": config.DanConfig(),
    "1500": config.DanConfig(),
    "2000": config.DanConfig()
}
root = "/"

if __name__ == '__main__':
    for k in DICT:
        cfgs = DICT[k](root)
        runner = baseline.BaselineDAN(cfgs)

