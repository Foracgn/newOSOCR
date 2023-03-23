import datetime

from torch.utils.data import DataLoader
from torch import nn
import torch

from task.common import loader, net, optimizer
from task.model import loss
from task.model.DAN import vis_dan

from neko_sdk.ocr_modules.trainable_losses.neko_url import neko_unknown_ranking_loss
from neko_sdk.ocr_modules.trainable_losses.cosloss import neko_cos_loss2
from neko_sdk.ocr_modules.neko_confusion_matrix import neko_confusion_matrix


class BaselineDAN:
    def __init__(self, cfgs):
        # 类内变量
        self.model = None

        self.trainLoader = None
        self.testLoader = None
        self.allLoader = None
        self.lossCounter = None
        self.testReject = None
        self.testAccuracy = None
        self.trainAccuracy = None
        self.standardCE = None
        self.optimizerSchedulers = None
        self.optimizers = None

        self.url = None
        self.cosLoss = None
        self.wmar = None
        self.wemb = None
        self.wsim = None
        self.wcls = None

        # 初始化
        self.cfgs = cfgs
        self.setupDataloader()
        self.setup()

        print("------------")
        print('prepare done')
        print("------------")

    def setup(self):
        self.model = net.loadNet(self.cfgs)
        self.setLoss()
        self.optimizers, self.optimizerSchedulers = optimizer.generateOptimizer(self.cfgs, self.model)

    def setupDataloader(self):
        if "datasetTrain" in self.cfgs.datasetConfigs:
            self.trainLoader, self.testLoader = loader.loadDataset(self.cfgs, DataLoader)
            self.setCounters()

    def setCounters(self):
        self.trainAccuracy = self.getAccuracy('train accuracy:', self.cfgs.datasetConfigs['trainCaseSensitive'])
        self.testAccuracy = self.getAccuracy('test accuracy:', self.cfgs.datasetConfigs['testCaseSensitive'])
        self.testReject = self.getReject('test reject:', self.cfgs.datasetConfigs['testCaseSensitive'])
        self.lossCounter = self.getLoss(self.cfgs.globalConfigs['showInterval'])

    def setLoss(self):
        self.standardCE = nn.CrossEntropyLoss().cuda()
        self.url = neko_unknown_ranking_loss()
        self.cosLoss = neko_cos_loss2().cuda()
        self.wcls = self.cfgs.lossWeight['wcls']
        self.wsim = self.cfgs.lossWeight['wsim']
        self.wemb = self.cfgs.lossWeight['wemb']
        self.wmar = self.cfgs.lossWeight['wmar']

    def run(self, datasetPath, measureReject):
        totalIter = len(self.trainLoader)
        for one in self.model:
            one.train()
        for nEpoch in range(0, self.cfgs.globalConfigs['epoch']):
            for i, batch in enumerate(self.trainLoader):
                self.trainIter(nEpoch, i, batch, totalIter)
            optimizer.UpdatePara(self.optimizerSchedulers, frozen=[])
            for i in range(0, len(self.model)):
                torch.save(
                    self.model[i].state_dict(),
                    self.cfgs.savingConfigs['savingPath'] + 'E{}_M{}.pth'.format(nEpoch, i)
                )

    def runTest(self, datasetPath, reject, debug=False):
        with torch.no_grad():
            tools = [self.testAccuracy, net.FlattenLabel]
            if reject:
                tools = [None, net.FlattenLabel, self.testReject]
            self.test(tools, miter=self.cfgs.globalConfigs['testMiter'], debug=debug, datasetPath=None)
            self.testAccuracy.clear()

    def test(self, tools, miter=1000, datasetPath=None, debug=False):
        net.TrainOrEval(self.model, 'Eval')
        proto, semblance, pLabel, tdict = self.model[3].dumpAll()
        counter = 0
        visualizer = None
        if datasetPath is not None:
            visualizer = vis_dan.VisDan(datasetPath)
        confusionMatrix = neko_confusion_matrix()

        for batch in self.testLoader:
            if counter > miter:
                break
            counter += 1
            image = batch['image']
            label = batch['label']
            target = self.model[3].encode(proto, pLabel, tdict, label)
            image = image.cuda()
            labelFlatten, length = tools[1](target)
            target.cuda()
            labelFlatten.cuda()
            features = self.model[0](image)
            one = self.model[1](features)
            output, outLength, one = self.model[2](features[-1], proto, semblance, pLabel, one, None, length, True)
            charOutput, predictProb = self.model[3].decode(output, outLength, proto, pLabel, tdict)
            tools[0].addIter(charOutput, outLength, label, debug)

            for i in range(len(charOutput)):
                confusionMatrix.addpairquickandugly(charOutput[i], label[i])

            if visualizer is not None:
                visualizer.addBatch(image, one, label, charOutput)

        if datasetPath is not None:
            confusionMatrix.save_matrix(datasetPath)
        tools[0].show()
        net.TrainOrEval(self.model, 'Train')

    def trainIter(self, nEpoch, idx, batch, tot):
        if 'cased' in batch:
            self.fpbp(batch['image'], batch['label'], batch['cased'])
        else:
            self.fpbp(batch['image'], batch['label'])

        for one in self.model:
            nn.utils.clip_grad_norm(one.parameters(), 20, 2)
        optimizer.UpdatePara(self.optimizers, frozen=[])
        if idx % self.cfgs.globalConfigs['showInterval'] == 0 and idx != 0:

            print(datetime.datetime.now().strftime('%H:%M:%S'))
            oneLoss, terms = self.lossCounter.getLossAndTerms()
            print('Epoch: {}, Iter: {}/{}, Loss dan: {}'.format(
                nEpoch,
                idx,
                tot,
                oneLoss)
            )
            if len(terms):
                print(terms)
            self.show()

        if nEpoch % self.cfgs.savingConfigs['savingEpochInterval'] == 0 and \
                idx % self.cfgs.savingConfigs['savingEpochInterval'] == 0 and idx != 0:
            for i in range(0, len(self.model)):
                torch.save(
                    self.model[i].state_dict(),
                    self.cfgs.savingConfigs['savingPath'][i]
                )

    def fpbp(self, image, label, cased=None):  # Forward Propagation And Backward Propagation
        proto, semblance, pLabel, tdict = self.makeProto(label)
        target = self.model[3].encode(proto, pLabel, tdict, label)

        net.TrainOrEval(self.model, 'Train')
        image = image.cuda()
        labelFlatten, length = net.FlattenLabel(target)
        target = target.cuda()
        labelFlatten = labelFlatten.cuda()
        net.ZeroGrad(self.model)
        features = self.model[0](image)
        one = self.model[1](features)

        # TODO train accuracy
        outCls, outCos = self.model[2](features[-1], proto, semblance, pLabel, one, target, length)
        charOutput, predictProb = self.model[3].decode(outCls, length, proto, pLabel, tdict)
        labels = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                  range(len(target))]
        self.trainAccuracy.addIter(charOutput, length, labels)
        protoLoss = nn.functional.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        res = torch.ones_like(torch.ones(outCls.shape[-1])).to(proto.device).float()
        res[-1] = 0.1
        CLSLoss = nn.functional.cross_entropy(outCls, labelFlatten, res)
        COSLoss = self.cosLoss(outCos, labelFlatten)
        marginLoss = self.url.forward(outCls, labelFlatten, 0.5)
        oneLoss = COSLoss * self.wsim + CLSLoss * self.wcls + marginLoss * self.wmar + self.wemb * protoLoss

        terms = {
            'total': oneLoss.detach().item(),
            'margin': marginLoss.detach().item(),
            "main": CLSLoss.detach().item(),
            "sim": COSLoss.detach().item(),
            "emb": protoLoss.detach().item(),
        }

        self.lossCounter.addIter(oneLoss, terms)
        oneLoss.backward()

    def show(self):
        self.trainAccuracy.show()
        self.trainAccuracy.clear()

    def makeProto(self, label):
        return self.model[3].sampleTrain(label)

    @staticmethod
    def getAccuracy(key, sensitive):
        return loss.AccuracyCounter(key, sensitive)

    @staticmethod
    def getReject(key, sensitive):
        return loss.RejectAccuracyCounter(key, sensitive)

    @staticmethod
    def getLoss(interval):
        return loss.LossCounter(interval)
