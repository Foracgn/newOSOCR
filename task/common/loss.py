from task.OCR.oracle.share_dict import oracleDict


cls_emb = ["CE", {
    "wcls": 1,
    "wace": 0,
    "wsim": 0,
    "wemb": 0.3,
    "wmar": 0,
}]


def TransUnknownLabel(label, testDict, complexLabel=False):
    allSetLabel = []
    if complexLabel:
        for one in label:
            allSetLabel.append(one if one in testDict else "⑨")
    else:
        for oneLabel in label:
            allSetLabel.append("".join([c if c in testDict else "⑨" for c in oneLabel]))
    return allSetLabel


def TransPredictLabel(targetLabel, predictLabel, complexLabel=False):
    if not complexLabel:
        return

    for one in targetLabel:
        print(one, end=" ")
    print()
    for one in targetLabel:
        if one in oracleDict:
            print(oracleDict[one], end=" ")
        else:
            print("⑨", end=" ")
    print()
    for one in predictLabel:
        print(one, end=" ")
    print()
    for one in predictLabel:
        if one in oracleDict:
            print(oracleDict[one], end=" ")
        else:
            print("⑨", end=" ")
    print()
    input()
