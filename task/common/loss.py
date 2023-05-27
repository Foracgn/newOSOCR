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


def TransPredictLabel(targetLabel, predictLabel, complexLabel=False, length=8):
    if not complexLabel:
        return

    print("=============target label============")
    for i in range(0, length):
        one = targetLabel[i]
        print(one, end=" ")
    print()
    for i in range(0, length):
        one = targetLabel[i]
        if one in oracleDict:
            print(oracleDict[one], end=" ")
        else:
            print("⑨", end=" ")
    print()
    print("============predict label============")
    for i in range(0, length):
        one = predictLabel[i]
        print(one, end=" ")
    print()
    for i in range(0, length):
        one = predictLabel[i]
        if one in oracleDict:
            print(oracleDict[one], end=" ")
        else:
            print("⑨", end=" ")
    print()
    input()
