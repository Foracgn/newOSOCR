import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import lmdb
import six
import image


class lmdbDataset(Dataset):
    def __init__(self, roots=None, ratio=None, imgH=32, imgW=128, transform=None, globalState='Test', maxT=25, repeat=1,
                 qhbAUG=False, forceTargetRatio=None):
        self.envs = []
        self.roots = []
        self.maxT = maxT
        self.numSamples = 0
        self.lengths = []
        self.ratio = []
        self.globalState = globalState
        self.repeat = repeat
        self.qhbAUG = qhbAUG
        self.setDataset(roots)

        self.transform = transform
        self.maxLen = len(self.lengths)
        self.imgH = imgH
        self.imgW = imgW

        if ratio is not None:
            assert len(roots) == len(ratio)
            for i in range(0, len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0, len(roots)):
                self.ratio.append(self.lengths[i]/float(self.numSamples))

        if forceTargetRatio is None:
            self.targetRatio = imgW / float(imgH)
        else:
            self.targetRatio = forceTargetRatio

    def __fromWhich__(self):
        res = random.random()
        tot = 0
        for i in range(0, len(self.ratio)):
            tot += self.ratio[i]
            if res <= tot:
                return i

    def __getitem__(self, index):
        fromWhich = self.__fromWhich__()
        if self.globalState == 'Train':
            index = random.randint(0, self.maxLen - 1)
        index = index % self.lengths[fromWhich]
        with self.envs[fromWhich].begin(write=False) as res:
            imgKey = 'image-%09d' % index

            imgBuff = res.get(imgKey.encode())
            buff = six.BytesIO()
            buff.write(imgBuff)
            buff.seek(0)
            img = image.open(buff)

            labelKey = 'label-%09d' % index
            label = str(res.get(labelKey.encode())).decode('utf-8')

            if len(label) > self.maxT - 1 and self.globalState == 'Train':
                print('sample too long')
                return self[index + 1]

            img, bMask = self.keepRatioResize(img.convert('RGB'))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            if self.transform:
                img = self.transform(img)
            sample = {
                'image': img,
                'label': label,
                'bmask': bMask,
            }
            return sample

    def keepRatioResize(self, img):
        curRatio = img.size[0] / float(img.size[1])
        maskH = self.imgH
        maskW = self.imgW
        img = np.array(img)

        if self.qhbAUG:
            img = 1
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if curRatio > self.targetRatio:
            curTargetH = self.imgH
            curTargetW = self.imgW
        else:
            curTargetH = self.imgH
            curTargetW = int(self.imgH * curRatio)
        img = cv2.resize(img, (curTargetW, curTargetH))
        startX = int((maskH - img.shape[0]) / 2)
        startY = int((maskW - img.shape[1]) / 2)
        mask = np.zeros([maskH, maskW]).astype(np.uint8)
        mask[startX:startX + img.shape[0], startY:startY + img.shape[1]] = img
        bMask = np.zeros([maskH, maskW]).astype(np.float)
        bMask[startX:startX + img.shape[0], startY:startY + img.shape[1]] = 1

        img = mask
        return img, bMask

    def setDataset(self, roots):
        for one in roots:
            env = lmdb.open(
                one,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            with env.begin(write=False) as res:
                samples = int(res.get('num-samples'.encode()))
            self.numSamples += samples
            self.lengths.append(samples)
            self.roots.append(one)
            self.envs.append(env)

    def __len__(self):
        return self.numSamples
