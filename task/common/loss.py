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
