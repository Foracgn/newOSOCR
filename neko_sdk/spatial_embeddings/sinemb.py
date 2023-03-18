import torch
from torch import nn
from neko_sdk.thirdparty.PositionalEncoding2D.positionalembedding2d import positionalencoding2d


class neko_sin_se(nn.Module):
    def __init__(self, dim=32):
        super(neko_sin_se, self).__init__()
        self.sew = -1
        self.seh = -1
        self.se = None
        self.dim = dim
        self.devinc = nn.Parameter(torch.tensor([0]), requires_grad=False)

    def forward(self, w, h):
        if w != self.sew or h != self.seh:
            self.seh = h
            self.sew = w
            self.se = positionalencoding2d(self.dim, h, w).to(self.devinc.device)
        return self.se.to(self.devinc.device)

# if __name__ == '__main__':
#     neko_sin_se(32)(12,12)
