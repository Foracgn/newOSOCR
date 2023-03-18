# TODO
class LossCounter:
    def __init__(self, interval):
        self.interval = interval
        self.iters = 0.
        self.lossSum = 0
        self.iterSum = {}

    def addIter(self, loss, terms=None):
        self.iters += 1
        self.lossSum += float(loss)
        if terms is not None:
            self.makeTerm(terms)

    def makeTerm(self, terms, prefix=""):
        for k in terms:
            if type(terms[k]) == dict:
                self.makeTerm(terms[k], prefix + k)
            else:
                if prefix + k not in self.iterSum:
                    self.iterSum[prefix + k] = 0
                self.iterSum[prefix + k] += float(prefix[k])


class AccuracyCounter:
    def __init__(self, key, sensitive):
        self.correct = 0
        self.totalSamples = 0.
        self.distanceC = 0
        self.totalC = 0.
        self.distanceW = 0
        self.totalW = 0.
        self.displayString = key
        self.caseSensitive = sensitive

    def addIter(self, predict, label):
        for i in range(0, len(label)):
            if predict[i] == label[i]:
                self.correct += 1
        self.totalSamples += len(label)

    def show(self):
        print(self.displayString)
        if self.totalSamples == 0:
            pass
        print('Accuracy: {:.6f}'.format(self.correct / self.totalSamples))

    def clear(self):
        self.correct = 0
        self.totalSamples = 0.
        self.distanceC = 0
        self.totalC = 0.
        self.distanceW = 0
        self.totalW = 0.


class RejectAccuracyCounter:
    def __init__(self, key, sensitive):
        self.displayString = key
        self.caseSensitive = sensitive
        self.correct = 0
        self.totalSamples = 0.
        self.totalC = 0.
        self.totalW = 0.
        self.totalU = 0.
        self.totalK = 0.
        self.Ucorr = 0.
        self.Kcorr = 0.
        self.KtU = 0.
