from torch import nn


class ConvolutionAlignment(nn.Module):
    def __init__(self, scales, maxT, depth, numChannel):
        super(ConvolutionAlignment, self).__init__()
        netRes = []
        # fpn: feature pyramid network
        for i in range(1, len(scales)):
            kSize = [3, 3, 5]
            rH = int(scales[i - 1][1] / scales[i][1])
            rW = int(scales[i - 1][2] / scales[i][2])
            kSizeH = 1 if scales[i - 1][1] == 1 else kSize[rH - 1]
            kSizeW = 1 if scales[i - 1][2] == 2 else kSize[rW - 1]
            netRes.append(
                nn.Sequential(
                    nn.Conv2d(
                        scales[i - 1][0], scales[i][0],
                        (kSizeH, kSizeW),
                        (rH, rW),
                        (int((kSizeH - 1) / 2), int((kSizeW - 1) / 2))
                    ),
                    nn.BatchNorm2d(scales[i][0]),
                    nn.ReLU(True)
                )
            )
        self.fpn = nn.Sequential(*netRes)
        inShape = scales[-1]
        strides = []
        kSizeConv = []
        kSizeDeConv = []
        h, w = inShape[1], inShape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth / 2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1]
            strides.append(stride)
            kSizeConv.append([3, 3])
            kSizeDeConv.append([_ ** 2 for _ in stride])

        convRes = [
            nn.Sequential(
                nn.Conv2d(
                    inShape[0], numChannel,
                    tuple(kSizeConv[0]),
                    tuple(strides[0]),
                    (int((kSizeConv[0][0] - 1) / 2), int((kSizeConv[0][1] - 1) / 2))
                ),
                nn.BatchNorm2d(numChannel),
                nn.ReLU(True)
            )
        ]
        self.conv = nn.Sequential(*convRes)

        deConvRes = []
        for i in range(1, int(depth / 2)):
            deConvRes.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        numChannel, numChannel,
                        tuple(kSizeDeConv[int(depth / 2) - i]),
                        tuple(strides[int(depth / 2) - i]),
                        (int(kSizeDeConv[int(depth / 2) - i][0] / 4.), int(kSizeDeConv[int(depth / 2) - i][1] / 4.))),
                    nn.BatchNorm2d(numChannel),
                    nn.ReLU(True)
                )
            )
        deConvRes.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    numChannel,
                    maxT,
                    tuple(kSizeDeConv[0]),
                    tuple(strides[0]),
                    (int(kSizeDeConv[0][0] / 4.), int(kSizeDeConv[0][1] / 4.))),
                nn.Sigmoid()
            )
        )
        self.deConv = nn.Sequential(*deConvRes)

    def forward(self, data):
        x = data[0]
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + data[i + 1]
        convFeats = []
        for i in range(0, len(self.conv)):
            x = self.conv[i](x)
            convFeats.append(x)
        for i in range(0, len(self.deConv) - 1):
            x = self.deConvs[i](x)
            f = convFeats[len(convFeats) - 2 - i]
            x = x[:, :, :f.shape[2], :f.shape[3]] + f
        x = self.deConv[-1](x)
        return x
