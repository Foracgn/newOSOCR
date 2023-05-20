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
    oracleDict = []
    testRoot = setConfigs['datasetRoot'] + setConfigs['oracle_test_out_set_dict']['root']

    for i in range(len(setConfigs['oracle_dict'])):
        taskPath = [
            modelRoot + setConfigs['basic']['FE'+str(i)],
            modelRoot + setConfigs['basic']['CAM'+str(i)],
            modelRoot + setConfigs['basic']['DTD'+str(i)],
            modelRoot + setConfigs['basic']['PE'+str(i)]
        ]
        modelPath.append(taskPath)
        oneDict = setConfigs['datasetRoot']+setConfigs['oracle_test_out_set_dict'+str(i)]
        oracleDict.append(oneDict)

    for i in range(len(modelPath)):
        cfgs = base.OracleDanConfig(
                modelPath[i],
                oracleDict[i],
                [testRoot],
                [testRoot],
                mode="Test"
        )

        runner = baseline.BaselineDAN(cfgs)
        runner.runTest(modelRoot, True, complexLabel=True)
        print("oracle task", i, "done")
