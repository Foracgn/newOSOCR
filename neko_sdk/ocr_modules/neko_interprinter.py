from torch import nn
from torchvision.models import resnet18, resnet34
from neko_sdk.encoders.ocr_networks.neko_pyt_resnet_np import resnet18np, resnet34np
# from neko_sdk.encoders.feat_networks.ires import conv_iResNet
import torch


class neko_visual_only_interprinter(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(neko_visual_only_interprinter, self).__init__()
        if core is None:
            self.core = resnet18(num_classes=feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"];
        vp = self.core(view_dict)

        # print(nvp.norm(dim=1))
        return vp


# class magic_core(nn.Module):
#     def __init__(this,feature_cnt):
#         super(magic_core, this).__init__();
#         this.c=conv_iResNet([3, 32, 32], [2, 2, 2, 2], [1, 2, 2, 2], [32, 32, 32, 32],
#                      init_ds=2, density_estimation=False, actnorm=True);
#         this.f=torch.nn.Linear(768,feature_cnt,False)
#         this.d=torch.nn.Dropout(0.1);
#     def forward(this,x):
#         c=this.c(x);
#         c=c.mean(dim=(2,3));
#         p=this.f(c);
#         return this.d(p)
#
class neko_visual_only_interprinter_inv(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(neko_visual_only_interprinter_inv, self).__init__()
        if core is None:
            self.core = magic_core(feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"];
        vp = self.core(view_dict)

        # print(nvp.norm(dim=1))
        return vp


class neko_visual_only_interprinterHD(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(neko_visual_only_interprinterHD, self).__init__()
        if core is None:
            self.core = resnet18np(outch=feature_cnt)
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"];
        vp = self.core(view_dict).permute(0, 2, 3, 1).reshape(view_dict.shape[0], -1)
        # print(nvp.norm(dim=1))
        return vp


class neko_visual_only_interprinterR34(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(neko_visual_only_interprinterR34, self).__init__();
        if core is None:
            self.core = resnet34(num_classes=feature_cnt);
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"];
        vp = self.core(view_dict)
        # print(nvp.norm(dim=1))
        return vp


class neko_structural_visual_only_interprinter(nn.Module):
    def __init__(self, feature_cnt, core=None):
        super(neko_structural_visual_only_interprinter, self).__init__()
        if core is None:
            self.core = resnet18np(outch=feature_cnt);
        else:
            self.core = core

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"];
        vp = self.core(view_dict)
        return vp.view(vp.shape[0], -1)
        # print(nvp.norm(dim=1))


# it has the core, but not it's prameter.
# it does the bp, but not the update.
# it is weird.
from neko_sdk.etc_modules.neko_att_ap import neko_att_ap
from torch import nn


class neko_weird_visual_only_interprinter(nn.Module):
    def __init__(self, feature_cnt, core):
        super(neko_weird_visual_only_interprinter, self).__init__()
        self.core = [core]
        self.compresser_att = neko_att_ap(feature_cnt)
        self.fc = nn.Linear(feature_cnt, feature_cnt)

    def forward(self, view_dict):
        # vis_proto=view_dict["visual"];
        self.core[0].train()
        vp = self.core[0](view_dict)[-1]
        vp = self.fc(self.compresser_att(vp))
        # print(nvp.norm(dim=1))
        return vp
