import torch


class neko_confusion_matrix:
    def __init__(self, charset=None):
        if charset is not None:
            self.charset = charset
        else:
            self.charset = set()

        self.edict = {}
        self.iddict = {}
        total = 0
        for k in self.charset:
            self.edict[k] = {}
            self.iddict[k] = total
            total += 1

    def addpairquickandugly(self, pred, gt):
        minlen = min(len(pred), len(gt))
        for i in range(minlen):
            pc = pred[i]
            gc = gt[i]
            if pc not in self.charset:
                self.charset.add(pc)
                self.edict[pc] = {}
                self.iddict[pc] = len(self.iddict)
            if gc not in self.charset:
                self.charset.add(gc)
                self.edict[gc] = {}
                self.iddict[gc] = len(self.iddict)

            if pc not in self.edict[gc]:
                self.edict[gc][pc] = 0
            self.edict[gc][pc] += 1

    def save_matrix(self, dst):
        torch.save(self.edict, dst)
