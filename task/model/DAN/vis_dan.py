import os
from torch import nn
import numpy as np
import cv2


def imageName(imagePath, counter):
    return os.path.join(imagePath, str(counter) + "_img.jpg")


def stackName(imagePath, counter):
    return os.path.join(imagePath, str(counter) + "_img.jpg")


def attName(imagePath, counter, t):
    return os.path.join(imagePath, str(counter) + "att_" + str(t) + ".jpg")


def resName(imagePath, counter):
    return os.path.join(imagePath, str(counter) + "_res.txt")


class VisDan:
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        os.makedirs(datasetPath, exist_ok=True)
        self.counter = 0

    def addBatch(self, batch, one, label, out):
        length = batch.shape[0]
        masks = nn.functional.interpolate(one, [batch.shape[2], batch.shape[3]], mode='bilinear')

        for i in range(length):
            self.write(self.counter, batch[i], masks[i], label[i], out[i])
            self.counter += 1

    def write(self, counter, oneBatch, mask, label, out):
        imagePath = imageName(self.datasetPath, counter)
        resPath = resName(self.datasetPath, counter)
        cv2.imwrite(imagePath, (oneBatch * 255)[0].detach().cpu().numpy().astype(np.uint8))
        with open(resPath, "w+") as fp:
            fp.write(label + "\n")
            fp.write(out + "\n")

        for i in range(min(len(label) + 3, mask.shape[0])):
            maskName = attName(self.datasetPath, id, i)
            cv2.imwrite(maskName,
                        (mask[i] * 200 * oneBatch + 56 * oneBatch)[0].detach().cpu().numpy().astype(np.uint8))

    def addImage(self, images, label, repCharOutput, charOutput, names):
        batchSize = images[0].shape[0]
        for i in range(batchSize):
            for j in range(len(images)):
                imagePath = imageName(self.datasetPath, str(self.counter) + names[j])
                cv2.imwrite(imagePath, (images[j][i] * 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
            self.writeImage(self.counter, label[i], charOutput[i], repCharOutput[i])
            self.counter += 1

    def writeImage(self, index, label, out, repOut):
        resPath = resName(self.datasetPath, index)
        with open(resPath, "w+") as fp:
            fp.write(label+"\n")
            fp.write(out+"\n")
