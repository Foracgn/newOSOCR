import datetime

from torch.utils.data import DataLoader
from torch import nn
import torch

from neko_sdk.AOF.neko_lens import vis_lenses
from task.common import loader, net, optimizer
from task.common.loss import TransUnknownLabel, TransPredictLabel
from task.model import loss
from task.model.DAN import vis_dan

from neko_sdk.ocr_modules.trainable_losses.neko_url import neko_unknown_ranking_loss
from neko_sdk.ocr_modules.trainable_losses.cosloss import neko_cos_loss2
from neko_sdk.ocr_modules.prototypers.neko_prototyper_core import neko_prototype_core_basic_shared


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
                    self.cfgs.savingConfigs['savingPath'][i]
                )

    def runTest(self, datasetPath, reject, debug=False, complexLabel=False):
        with torch.no_grad():
            tools = [self.testAccuracy, net.FlattenLabel]
            if reject:
                tools = [self.testReject, net.FlattenLabel]
            self.test(tools, miter=self.cfgs.globalConfigs['testMiter'], datasetPath=None, reject=reject, debug=debug, complexLabel=complexLabel)
            self.testAccuracy.clear()

    def test(self, tools, miter=1000, datasetPath=None, reject=False, debug=False, complexLabel=False):
        net.TrainOrEval(self.model, 'Eval')

        testMeta = None
        if "testMeta" in self.cfgs.datasetConfigs:
            testMeta = torch.load(self.cfgs.datasetConfigs["testMeta"])

        if testMeta is None:
            proto, semblance, pLabel, testDict = self.model[3].dumpAll()
            # model[3].dumpAll == model[3].dwcore.dump_all()
            # 目前无用分支
        else:
            testCore = neko_prototype_core_basic_shared(testMeta, self.model[3].dwcore)
            testCore.cuda()
            proto, pLabel, testDict = testCore.dump_all()
            semblance = None

        counter = 0
        allLength = 0
        visualizer = None
        if datasetPath is not None:
            visualizer = vis_dan.VisDan(datasetPath)
        for batch in self.testLoader:
            if counter > miter:
                break
            counter += 1
            image = batch['image']
            label = batch['label']
            target = self.model[3].encode(testDict, label)

            allLength += image.shape[0]
            image = image.cuda()
            target = target.cuda()
            labelFlatten, labelLength = tools[1](target)

            features, grid = self.model[0](image)
            one = self.model[1](features)
            output, outLength, one = self.model[2](features[-1], proto, pLabel, one, None, labelLength, True)
            charOutput, predictProb = self.model[3].decode(output, outLength, testDict)

            # repCharOutput = [[i] for i in charOutput]
            # reject
            if reject:
                allSetLabel = TransUnknownLabel(label, testDict, complexLabel)
                tools[0].addIter(charOutput, outLength, allSetLabel, debug)
            else:
                tools[0].addIter(charOutput, outLength, label, debug)
                TransPredictLabel(label, charOutput, complexLabel)
            if datasetPath:
                features, grid = self.model[0](image)
                if len(grid) > 0:
                    resData = vis_lenses(image, grid)[1]
                    predictList = [[i] for i in charOutput]
                    if visualizer is not None:
                        visualizer.addImage([image, resData], label, predictList, charOutput, ["before", "after"])
        tools[0].show()
        print(all)
        net.TrainOrEval(self.model, 'Train')

    def trainIter(self, nEpoch, idx, batch, tot):
        # run NET DAN
        if 'cased' in batch:
            self.fpbp(batch['image'], batch['label'], batch['cased'])
        else:
            self.fpbp(batch['image'], batch['label'])

        for one in self.model:
            nn.utils.clip_grad_norm(one.parameters(), 20, 2)
        optimizer.UpdatePara(self.optimizers, frozen=[])
        # print
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
        # make proto -> core.sample_charset_by_text
        proto, _, pLabel, tdict = self.makeProto(label)
        target = self.model[3].encode(tdict, label)
        labelFlatten, length = net.FlattenLabel(target)
        target = target.cuda()
        labelFlatten = labelFlatten.cuda()
        targetLabels = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                        range(len(target))]

        net.TrainOrEval(self.model, 'Train')
        image = image.cuda()
        net.ZeroGrad(self.model)
        features, grid = self.model[0](image)
        one = self.model[1](features)

        outCls, _ = self.model[2](features[-1], proto, pLabel, one, target, length)
        # out, attention

        charOutput, predictProb = self.model[3].decode(outCls, length, tdict)
        # 训练中预测结果
        self.trainAccuracy.addIter(charOutput, length, targetLabels)

        protoLoss = nn.functional.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean()
        res = torch.ones_like(torch.ones(outCls.shape[-1])).to(proto.device).float()
        res[-1] = 0.1
        CLSLoss = nn.functional.cross_entropy(outCls, labelFlatten, res)
        SimLoss = self.cosLoss(_, labelFlatten)
        marginLoss = self.url.forward(outCls, labelFlatten, 0.5)
        # unknown ranking loss
        oneLoss = SimLoss * self.wsim + CLSLoss * self.wcls + marginLoss * self.wmar + self.wemb * protoLoss

        terms = {
            'total': oneLoss.detach().item(),
            'margin': marginLoss.detach().item(),
            "main": CLSLoss.detach().item(),
            "sim": SimLoss.detach().item(),
            "emb": protoLoss.detach().item(),
        }

        self.lossCounter.addIter(oneLoss, terms)
        oneLoss.backward()

    def show(self):
        self.trainAccuracy.show()
        self.trainAccuracy.clear()

    def makeProto(self, label):
        return self.model[3].sampleTrain(label)

    def transMasterLabel(self, label):
        return self.model[3].dwcore.tranLabel(label)

    @staticmethod
    def getAccuracy(key, sensitive):
        return loss.AccuracyCounter(key, sensitive)

    @staticmethod
    def getReject(key, sensitive):
        return loss.RejectAccuracyCounter(key, sensitive)

    @staticmethod
    def getLoss(interval):
        return loss.LossCounter(interval)
