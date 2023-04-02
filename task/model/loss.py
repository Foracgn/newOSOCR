import editdistance as ed


class LossCounter:
    def __init__(self, interval):
        self.interval = interval
        self.iters = 0.
        self.lossSum = 0
        self.termSum = {}

    def show(self):
        print(self.getLossAndTerms())

    def clear(self):
        self.iters = 0.
        self.lossSum = 0
        self.termSum = {}

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
                if prefix + k not in self.termSum:
                    self.termSum[prefix + k] = 0
                self.termSum[prefix + k] += float(terms[k])

    def getLoss(self):
        if self.iters > 0:
            loss = self.lossSum / self.iters
        else:
            loss = 0
        return loss

    def getLossAndTerms(self):
        loss = self.getLoss()
        terms = {}
        for i in self.termSum:
            term = self.termSum[i] / self.iters if self.iters > 0 else 0
            terms[i] = term
        self.clear()
        return loss, terms


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

    def addIter(self, predict, length, label, debug=False):
        if label is None:
            return

        self.totalSamples += len(label)
        for i in range(0, len(predict)):
            if not self.caseSensitive:
                predict[i] = predict[i].lower().replace("⑨", "")
                label[i] = label[i].lower()
            all_words = []
            for w in label[i].split('|sadhkjashfkjasyhf') + predict[i].split('||sadhkjashfkjasyhf'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in label[i].split('||sadhkjashfkjasyhf')]
            p_words = [all_words.index(_) for _ in predict[i].split('||sadhkjashfkjasyhf')]
            self.distanceC += ed.eval(label[i], predict[i])
            self.distanceW += ed.eval(l_words, p_words)
            self.totalC += len(label[i])
            self.totalW += len(l_words)
            self.correct = self.correct + 1 if label[i] == predict[i] else self.correct

    def show(self):
        print(self.displayString)
        if self.totalSamples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.totalSamples,
            1 - self.distanceC / max(1.0, self.totalC),
            self.distanceC / max(1.0, self.totalC),
            self.distanceW / max(1.0, self.totalW)))

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

    def addIter(self, predict, labelLength, label, debug=False):
        if label is None:
            return

        self.totalSamples += len(label)
        for i in range(0, len(predict)):
            if not self.caseSensitive:
                predict[i] = predict[i].lower()
                label[i] = label[i].lower()
            allWords = []
            for w in label[i].split('|sadhkjashfkjasyhf') + predict[i].split('||sadhkjashfkjasyhf'):
                if w not in allWords:
                    allWords.append(w)
            l_words = [allWords.index(_) for _ in label[i].split('||sadhkjashfkjasyhf')]
            self.totalC += len(label[i])
            self.totalW += len(l_words)
            cflag = int(label[i] == predict[i])
            self.correct = self.correct + cflag
            if label[i].find("⑨") != -1:
                self.totalU += 1
                self.Ucorr += (predict[i].find("⑨") != -1)
            else:
                self.totalK += 1
                self.Kcorr += cflag
                # TODO KtU添加确认
                self.KtU += (predict[i].find("⑨") != -1)

    def show(self):
        print(self.displayString)
        if self.totalSamples == 0:
            pass
        R = self.Ucorr / self.totalU
        P = self.Ucorr / (self.Ucorr + self.KtU)
        F = 2 * (R * P) / (R + P)
        print("KACR: {:.6f},URCL:{:.6f}, UPRE {:.6f}, F {:.6f}".format(
            self.Kcorr / self.totalK,
            R, P, F
        ))
