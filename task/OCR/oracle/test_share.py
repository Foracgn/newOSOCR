from __future__ import print_function
from task.model.DAN import base
from task.framework import baseline
import yaml

constPath = "./task/yaml/oracle.yaml"
netNumber = 3

if __name__ == '__main__':
    data = open(constPath, 'r').read()
    setConfigs = yaml.load(data, Loader=yaml.FullLoader)
    modelRoot = setConfigs['modelRoot']
    modelPath = []
    oracleDict = []
    testRoot = setConfigs['datasetRoot'] + setConfigs['oracle_share']['root']

    taskPath = [
        modelRoot + setConfigs['basic']['FE'+str(netNumber)],
        modelRoot + setConfigs['basic']['CAM'+str(netNumber)],
        modelRoot + setConfigs['basic']['DTD'+str(netNumber)],
        modelRoot + setConfigs['basic']['PE'+str(netNumber)]
    ]
    modelPath.append(taskPath)
    oneDict = setConfigs['datasetRoot']+setConfigs['oracle_share']['dict']

    for i in range(len(modelPath)):
        cfgs = base.OracleDanConfig(
                modelPath[0],
                oneDict,
                [testRoot],
                [testRoot],
                mode="Test"
        )

        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(modelRoot, False, complexLabel=True)
        print("oracle task", i, "done")
