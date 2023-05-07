from __future__ import print_function
from task.model.DAN import base
from task.framework import baseline
import yaml

constPath = "./task/yaml/oracle.yaml"

if __name__ == '__main__':
    data = open(constPath, 'r').read()
    setConfigs = yaml.load(data, Loader=yaml.FullLoader)
    modelRoot = setConfigs['modelRoot']
    modelPath = []
    for i in range(0, 4):
        taskPath = [
            modelRoot + setConfigs['basic']['FE' + str(i)],
            modelRoot + setConfigs['basic']['CAM' + str(i)],
            modelRoot + setConfigs['basic']['DTD'+str(i)],
            modelRoot + setConfigs['basic']['PE'+str(i)]
        ]
        modelPath.append(taskPath)

    trainDict = ""
    testDict = setConfigs['datasetRoot']+setConfigs['oracle_dict']
    testRoot = setConfigs['datasetRoot']+setConfigs['oracle_root']

    cfgs = base.FreeDictDanConfig(
            modelPath[0],
            testDict,
            [testRoot],
            [testRoot],
            mode="Test"
    )

    runner = baseline.BaselineDAN(cfgs)
    runner.runTest(modelRoot, True)

    print("oracle task done")
