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
    taskPath = [
        modelRoot + setConfigs['basic']['FE0'],
        modelRoot + setConfigs['basic']['CAM0'],
        modelRoot + setConfigs['basic']['DTD0'],
        modelRoot + setConfigs['basic']['PE0']
    ]
    modelPath.append(taskPath)

    oracleDict = setConfigs['datasetRoot']+setConfigs['oracle_dict']
    oracleRoot = setConfigs['datasetRoot']+setConfigs['oracle_root']

    cfgs = base.OracleDanConfig(
            modelPath[0],
            oracleDict,
            [oracleRoot],
            [oracleRoot],
            mode="Test"
    )

    runner = baseline.BaselineDAN(cfgs)
    runner.runTest(modelRoot, True)

    print("oracle task done")
