class LossCounter:
    def __init__(self, interval):
        self.interval = interval
        self.iters = 0.
        self.lossSum = 0
        self.iterSum = {}


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
