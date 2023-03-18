import torch
from torch import nn
import random
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter, \
    neko_structural_visual_only_interprinter, neko_weird_visual_only_interprinter, \
    neko_visual_only_interprinterR34
from neko_sdk.ocr_modules.prototypers.neko_visual_center_prototyper import neko_abstract_visual_center_prototyper
import numpy as np
from collections import deque as python_queue
import regex

# this class defines how samples are sampled ^_^


class neko_prototype_core_basic(nn.Module):
    # every thing does not involve sampling
    PROTOENGINE = neko_visual_only_interprinter

    def make_proto_engine(self, meta, backbone, preload_tensor=None):
        self.proto_engine = self.PROTOENGINE(self.output_channel, backbone)
        self.prototype_cnt = -1
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        if meta is not None:
            self.masters = meta["master"]

            # Foes includes the characters looks like each other
            # but never share labels (They may actually have linguistic relationships...
            # Like yanderes in a broken relationship[x]).
            # This set helps implement ohem like minibatching on the huge labelset.
            # e.g. 'u' and 'ü'
            self.foes = meta["foes"]
            self.servants = meta["servants"]
            # union set of friend, harem and foe.
            self.related_proto_ids = meta["relationships"]

        self.sp_protos = torch.nn.Parameter(torch.rand([
            self.sp_cnt, self.output_channel]).float() * 2 - 1)
        self.register_parameter("sp_proto", self.sp_protos)

    def setup_common(self, output_channel,
                     meta,
                     backbone=None,
                     preload_tensor=None,
                     dropout=None):
        self.output_channel = output_channel
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size it's nightmare.
        self.dev_ind = torch.nn.Parameter(torch.rand([1]))
        list_character = list(meta["chars"])
        self.aligned_characters = meta["achars"]
        # characters without shape is generally what you do now want to sample.
        self.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        self.character = list(meta["sp_tokens"]) + list_character
        self.label_dict = meta["label_dict"]
        self.shaped_ids = set([self.label_dict[i] for i in self.shaped_characters])
        self.sp_cnt = len(meta["sp_tokens"])
        self.sp_tokens = meta["sp_tokens"]
        if dropout is not None:
            self.drop = torch.nn.Dropout(p=0.3)
        else:
            self.drop = None
        unk = self.label_dict["[UNK]"]
        # if the dict does not provide an specific unk token, set it to -1;
        for i, char in enumerate(self.character):
            # print(i, char)
            self.label_dict[char] = i
        # shapeless unk shall be excluded
        if unk < 0:
            self.label_set = set(self.label_dict.values()) - {unk}
        else:
            self.label_set = set(self.label_dict.values())

        if preload_tensor is None:
            self.norm_protos = meta["protos"][self.sp_cnt:]
        else:
            self.norm_protos = torch.load(preload_tensor)

        for i in range(len(self.norm_protos)):
            if self.norm_protos[i] is not None and self.norm_protos[i].max() > 20:
                self.norm_protos[i] = (self.norm_protos[i] - 127.5) / 128

        self.make_proto_engine(meta, backbone, preload_tensor=None)

    # defines sampler
    def setup_sampler(self, sampler_args):
        if sampler_args is None:
            masters_share = False
            max_match_size = 512
            val_frac = 0.8
            neg_servant = True
        else:
            masters_share = sampler_args["master_share"]
            max_match_size = sampler_args["max_batch_size"]
            val_frac = sampler_args["val_frac"]
            neg_servant = sampler_args["neg_servant"]
        self.masters_share = masters_share
        self.max_batch_size = max_match_size
        self.val_frac = val_frac
        self.neg_servant = neg_servant

    def __init__(self,
                 output_channel,
                 meta,
                 backbone=None,
                 preload_tensor=None,
                 sampler_args=None,
                 dropout=None,
                 ):
        self.foes = None
        self.servants = None
        self.related_proto_ids = None
        self.masters = None
        self.sp_protos = None
        self.label_dict = None
        self.shaped_characters = None
        self.character = None
        self.shaped_ids = None
        self.aligned_characters = None
        self.output_channel = None
        self.dev_ind = None
        self.sp_cnt = None
        self.sp_tokens = None
        self.drop = None
        self.label_set = None
        self.norm_protos = None
        self.neg_servant = None
        self.val_frac = None
        self.max_batch_size = None
        self.masters_share = None
        self.proto_engine = None
        self.prototype_cnt = None
        print("DEBUG-SDFGASDFGSDGASFGSD", dropout)
        super(neko_prototype_core_basic, self).__init__()
        self.setup_common(output_channel, meta, backbone, preload_tensor, dropout)
        self.setup_sampler(sampler_args)

    def debug(self, normpids, labels):
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in normpids]
        protos = ((torch.cat(normprotos, dim=-1).squeeze(0).squeeze(0) + 1) * 127.5).detach().cpu().numpy().astype(
            np.uint8)
        import cv2
        cv2.imshow(labels, protos[:, :32 * 32])
        cv2.waitKey(0)

    def get_protos(self, sppids, normpids):
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in normpids]
        spprotos = [self.sp_protos[i].unsqueeze(0) for i in sppids]
        normprotos = self.proto_engine(torch.cat(normprotos).repeat(1, 3, 1, 1).to(self.dev_ind.device))
        allproto = torch.cat(spprotos + [normprotos])
        if self.drop:
            allproto = self.drop(allproto)

        return allproto / torch.norm(allproto, dim=-1, keepdim=True)

    def get_plabel_and_dict(self, sappids, normpids):
        all_ids = sappids.union(normpids)
        new_id = 0
        plabels = []
        labmap = {}
        bidict = {}
        for i in all_ids:
            cha = self.aligned_characters[i]
            if self.masters_share:
                vlab = self.masters[i]
            else:
                vlab = i
            if vlab not in labmap:
                labmap[vlab] = new_id
                # A new label
                new_id += 1
            alab = labmap[vlab]
            plabels.append(alab)
            bidict[alab] = cha
            bidict[cha] = alab
        plabels.append(new_id)
        bidict["[UNK]"] = new_id
        bidict[new_id] = "⑨"

        return torch.tensor(plabels), bidict

    def grab_cluster(self, ch):
        chid = self.label_dict[ch]
        ret = {chid}
        if self.masters_share:
            ret.add(self.masters[chid])
            ret = ret.union(self.servants[self.masters[chid]])
        return ret

    def get_sampled_ids(self, plain_chars_in_data):
        cntval = int(len(plain_chars_in_data) * self.val_frac)
        cntval = min(self.max_batch_size - self.sp_cnt, cntval)
        trchs = set()
        related_chars_in_data = set()
        random.shuffle(plain_chars_in_data)
        # make sure no missing centers--
        # or it may enforce "A" to look like "a" encoded by proto CNN
        remaining = cntval
        for ch in plain_chars_in_data:
            if ch not in self.label_dict:
                continue
            new = self.grab_cluster(ch)
            ns = trchs.union(new)
            related_chars_in_data = related_chars_in_data.union(new)
            delta = len(ns) - len(trchs)
            if delta <= remaining:
                trchs = ns
                remaining -= delta
        remaining = self.max_batch_size - self.sp_cnt - len(trchs)
        plain_charid_not_in_data = list(self.shaped_ids - related_chars_in_data)
        random.shuffle(plain_charid_not_in_data)
        for chid in plain_charid_not_in_data:
            if chid not in trchs:
                if remaining == 0:
                    break
                if self.neg_servant is False and self.masters[chid] != chid:
                    continue
                remaining -= 1
                trchs.add(chid)

        trsps = set([self.label_dict[i] for i in self.sp_tokens])
        return trsps, trchs

    def sample_charset_by_text(self, text_batch):
        b = ""
        for _ in text_batch:
            b += _
        plain_chars_in_data = list(set(regex.findall(r'\X', b, regex.U)))
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        protos = self.get_protos(trsps, trchs)
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return protos, plabels, tdicts

    def dump_all(self):
        trsps = set([self.label_dict[i] for i in self.sp_tokens])
        trchs = set([self.label_dict[i] for i in self.shaped_characters])

        protos = self.get_protos(trsps, trchs)
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return protos, plabels, tdicts


class neko_prototype_core_basic_shared(nn.Module):

    def make_proto_engine(self, meta, core):
        self.proto_engine = core.proto_engine
        self.prototype_cnt = -1
        # handles Capitalize like problem.
        # In some annotations they have same labels, while in others not.
        # i.e, 'A' and 'a' can have the same label 'a',
        # '茴','回','囘' and '囬' can have the same label '回'
        self.masters = meta["master"]
        # Foes includes the characters looks like each other
        # but never share labels (They may actually have linguistic relationships...
        # Like yanderes in a broken relationship[x]).
        # This set helps implement ohem like minibatching on the huge labelset.
        # e.g. 'u' and 'ü'
        self.foes = meta["foes"]
        self.servants = meta["servants"]
        # union set of friend, harem and foe.
        self.related_proto_ids = meta["relationships"]

        self.sp_protos = core.sp_protos

    def setup_common(self,
                     meta,
                     core):
        # 原型空间初始化
        # meta内数据结构
        self.masters_share = True
        self.output_channel = core.output_channel
        # Generally, CUDA devices works well with dynamic batch size.
        # However, for Rocm devices it bites as it compiles kernel
        # everytime we change batch size it's nightmare.
        self.dev_ind = torch.nn.Parameter(torch.rand([1]))
        list_character = list(meta["chars"])
        self.aligned_characters = meta["achars"]
        # characters without shape is generally what you do now want to sample.
        self.shaped_characters = set(meta["chars"])
        # UNK is not a sp_token as it is centerless.
        self.character = list(meta["sp_tokens"]) + list_character
        self.label_dict = meta["label_dict"]
        self.shaped_ids = set([self.label_dict[i] for i in self.shaped_characters])
        # char的id映射
        self.sp_cnt = len(meta["sp_tokens"])
        self.sp_tokens = meta["sp_tokens"]
        self.drop = core.drop
        unk = self.label_dict["[UNK]"]
        # if the dict does not provide an specific unk token, set it to -1;
        # 去除dict里unk的部分
        for i, char in enumerate(self.character):
            # print(i, char)
            self.label_dict[char] = i
        # shapeless unk shall be excluded
        if unk < 0:
            self.label_set = set(self.label_dict.values()) - {unk}
        else:
            self.label_set = set(self.label_dict.values())

        self.norm_protos = meta["protos"][self.sp_cnt:]

        for i in range(len(self.norm_protos)):
            if self.norm_protos[i] is not None and self.norm_protos[i].max() > 20:
                self.norm_protos[i] = (self.norm_protos[i] - 127.5) / 128

        self.make_proto_engine(meta, core)

    def __init__(self,
                 meta,
                 core
                 ):
        super(neko_prototype_core_basic_shared, self).__init__()
        self.proto_engine = None
        self.prototype_cnt = None
        self.masters = None
        self.foes = None
        self.servants = None
        self.related_proto_ids = None
        self.sp_protos = None
        self.dev_ind = None
        self.aligned_characters = None
        self.shaped_characters = None
        self.character = None
        self.label_dict = None
        self.shaped_ids = None
        self.sp_cnt = None
        self.sp_tokens = None
        self.label_set = None
        self.norm_protos = None
        self.output_channel = None
        self.drop = None
        self.masters_share = None
        self.setup_common(meta, core)

    def debug(self, normpids, labels):
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in normpids]
        protos = ((torch.cat(normprotos, dim=-1).unsqueeze(0).unsqueeze(1) + 1) * 127.5).detach().cpu().numpy().astype(
            np.uint8)
        import cv2
        cv2.imshow("protos", protos)
        cv2.waitKey(0)

    def get_protos(self, sppids, normpids):
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in normpids]
        spprotos = [self.sp_protos[i].unsqueeze(0) for i in sppids]
        normprotos = self.proto_engine(torch.cat(normprotos).repeat(1, 3, 1, 1).to(self.dev_ind.device))
        allproto = torch.cat(spprotos + [normprotos])
        if self.drop:
            allproto = self.drop(allproto)

        return allproto / torch.norm(allproto, dim=-1, keepdim=True)

    def get_plabel_and_dict(self, sappids, normpids):
        all_ids = sappids.union(normpids)
        new_id = 0
        plabels = []
        labmap = {}
        bidict = {}
        for i in all_ids:
            cha = self.aligned_characters[i]
            if self.masters_share:
                vlab = self.masters[i]
            else:
                vlab = i
            if vlab not in labmap:
                labmap[vlab] = new_id
                # A new label
                new_id += 1
            alab = labmap[vlab]
            plabels.append(alab)
            bidict[alab] = cha
            bidict[cha] = alab
        plabels.append(new_id)
        bidict["[UNK]"] = new_id
        bidict[new_id] = "⑨"

        return torch.tensor(plabels), bidict

    def dump_all(self):
        trsps = set([self.label_dict[i] for i in self.sp_tokens])
        trchs = set([self.label_dict[i] for i in self.shaped_characters])

        protos = self.get_protos(trsps, trchs)
        plabels, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return protos, plabels, tdicts


class neko_prototype_core_structural(neko_prototype_core_basic):
    PROTOENGINE = neko_structural_visual_only_interprinter


class neko_prototype_core_structuralR34(neko_prototype_core_basic):
    PROTOENGINE = neko_visual_only_interprinterR34


class neko_prototype_core_weird(neko_prototype_core_basic):
    PROTOENGINE = neko_weird_visual_only_interprinter

    def get_protos(self, sppids, normpids):
        normprotos = [self.norm_protos[i - self.sp_cnt] for i in normpids]
        spprotos = [self.sp_protos[i].unsqueeze(0) for i in sppids]
        normprotos = self.proto_engine(torch.cat(normprotos).to(self.dev_ind.device))
        allproto = torch.cat(spprotos + [normprotos])
        if self.drop:
            allproto = self.drop(allproto)

        return allproto / torch.norm(allproto, dim=-1, keepdim=True)
