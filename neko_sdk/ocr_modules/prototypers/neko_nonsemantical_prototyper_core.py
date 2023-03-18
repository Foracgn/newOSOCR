import torch
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter
# this class defines how samples are sampled ^_^
from neko_sdk.ocr_modules.prototypers.neko_prototyper_core import neko_prototype_core_basic

import regex
import random


class neko_nonsematical_prototype_core_basic(neko_prototype_core_basic):
    # every thing does not involve sampling
    PROTOENGINE = neko_visual_only_interprinter

    def __init__(self,
                 output_channel,
                 meta,
                 backbone=None,
                 preload_tensor=None,
                 sampler_args=None,
                 dropout=None,
                 ):
        super().__init__(output_channel, meta, backbone, preload_tensor, sampler_args, dropout)
        self.label_set = None
        self.drop = None
        self.dev_ind = None
        self.output_channel = None
        self.norm_protos = None
        self.sp_cnt = None
        self.shaped_ids = None
        self.label_dict = None
        self.character = None
        self.shaped_characters = None
        self.aligned_characters = None
        self.sp_tokens = None

    def arm_meta(self, meta, preload_tensor):
        if meta is None:
            return

        list_character = list(meta["chars"])
        self.aligned_characters = meta["achars"]
        # characters without shape is generally what you always want to keep.
        self.shaped_characters = sorted(set(meta["chars"]))
        # UNK is not a sp_token as it is centerless.
        self.character = list(meta["sp_tokens"]) + list_character
        self.label_dict = meta["label_dict"]
        self.shaped_ids = set([self.label_dict[i] for i in self.shaped_characters])
        self.sp_cnt = len(meta["sp_tokens"])
        self.sp_tokens = meta["sp_tokens"]
        if preload_tensor is None:
            self.norm_protos = meta["protos"][self.sp_cnt:]
        else:
            self.norm_protos = torch.load(preload_tensor)

    def arm_none_meta(self):
        self.label_dict = {"[s]": 0, "[UNK]": 1}
        self.sp_cnt = 1

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
        if dropout is not None:
            self.drop = torch.nn.Dropout(p=0.3)
        else:
            self.drop = None

        if meta is not None:
            self.arm_meta(meta, preload_tensor)
        else:
            self.arm_none_meta()
            self.make_proto_engine(meta, backbone, preload_tensor=None)
            return
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

        for i in range(len(self.norm_protos)):
            if self.norm_protos[i] is not None and self.norm_protos[i].max() > 20:
                self.norm_protos[i] = (self.norm_protos[i] - 127.5) / 128

        self.make_proto_engine(meta, backbone, preload_tensor=None)

    def get_plabel_and_dict_core(self, sappids, normpids, masters_share):
        all_ids = sappids + normpids
        new_id = 0
        plabels = []
        labmap = {}
        bidict = {}
        sembs = []
        for i in all_ids:
            cha = self.aligned_characters[i]
            if masters_share:
                vlab = self.masters[i]
            else:
                vlab = i
            if vlab not in labmap:
                labmap[vlab] = new_id
                # A new label
                new_id += 1
                # sembs.append(this.semantic_embedding[vlab]);
            alab = labmap[vlab]
            plabels.append(alab)
            bidict[alab] = cha
            bidict[cha] = alab
        plabels.append(new_id)
        bidict["[UNK]"] = new_id
        # Well it has to be something --- at least ⑨ is alignment friendly and not likely to appear in the dataset
        bidict[new_id] = "⑨"
        return torch.tensor(plabels), None, bidict

    def get_plabel_and_dict(self, sappids, normpids):
        return self.get_plabel_and_dict_core(sappids, normpids, self.masters_share)

    def sample_charset_by_text_both(self, text_batch):
        b = ""
        for _ in text_batch:
            b += _

        plain_chars_in_data = list(set(regex.findall(r'\X', b, regex.U)))
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        trsps = list(trsps)
        protos = self.get_protos(trsps, trchs)
        plabels_cased, sembs_cased, tdicts_cased = self.get_plabel_and_dict_core(trsps, trchs, False)
        plabels_uncased, sembs_uncased, tdicts_uncased = self.get_plabel_and_dict_core(trsps, trchs, True)

        # this.debug(trchs,"meow");
        return protos, [sembs_uncased, sembs_cased], [plabels_uncased, plabels_cased], [tdicts_uncased, tdicts_cased]

    def sample_charset_by_text(self, text_batch):
        b = ""
        for _ in text_batch:
            b += _

        plain_chars_in_data = list(set(regex.findall(r'\X', b, regex.U)))
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        trsps = list(trsps)
        protos = self.get_protos(trsps, trchs)
        plabels, sembs, tdicts = self.get_plabel_and_dict(trsps, trchs)
        # this.debug(trchs,"meow");
        return protos, sembs, plabels, tdicts

    def dump_all(self):
        trsps = [self.label_dict[i] for i in self.sp_tokens]
        trchs = [self.label_dict[i] for i in self.shaped_characters]

        protos = self.get_protos(trsps, trchs)
        plabels, sembs, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return protos, sembs, plabels, tdicts


# g2 supports multiple
class neko_nonsematical_prototype_core_basic_g2rand(neko_nonsematical_prototype_core_basic):
    def mvn(self):
        for i in range(len(self.norm_protos)):
            if self.norm_protos[i] is not None and self.norm_protos[i][0].max() > 20:
                self.norm_protos[i] = [(_ - 127.5) / 128 for _ in self.norm_protos[i]]

    def get_protos_rand(self, sppids, normpids):
        normprotos = [random.choice(self.norm_protos[i - self.sp_cnt]) for i in normpids]
        # im = (torch.cat(normprotos[:16], 2)[0, 0] * 127 + 128).cpu().numpy().astype(np.uint8);
        # cv2.imshow( "a",im);
        # cv2.waitKey(0);
        spprotos = [self.sp_protos[i].unsqueeze(0) for i in sppids]
        normprotos = self.proto_engine(torch.cat(normprotos).repeat(1, 3, 1, 1).to(self.dev_ind.device))
        allproto = torch.cat(spprotos + [normprotos])
        if self.drop:
            allproto = self.drop(allproto)
        return allproto / torch.norm(allproto, dim=-1, keepdim=True)

    def get_protos_idx(self, sppids, normpids, idx):
        normprotos = [self.norm_protos[i - self.sp_cnt][idx] for i in normpids]
        spprotos = [self.sp_protos[i].unsqueeze(0) for i in sppids]
        normprotos = self.proto_engine(torch.cat(normprotos).repeat(1, 3, 1, 1).to(self.dev_ind.device))
        allproto = torch.cat(spprotos + [normprotos])
        if self.drop:
            allproto = self.drop(allproto)
        return allproto / torch.norm(allproto, dim=-1, keepdim=True)

    def sample_charset_by_text(self, text_batch):
        b = ""
        for _ in text_batch:
            b += _

        plain_chars_in_data = list(set(regex.findall(r'\X', b, regex.U)))
        trsps, trchs = self.get_sampled_ids(plain_chars_in_data)
        trchs = list(trchs)
        trsps = list(trsps)
        protos = self.get_protos_rand(trsps, trchs)
        plabels, sembs, tdicts = self.get_plabel_and_dict(trsps, trchs)
        # this.debug(trchs,"meow");
        return protos, sembs, plabels, tdicts

    def dump_all(self, rot=0, idx=0):
        trsps = [self.label_dict[i] for i in self.sp_tokens]
        trchs = [self.label_dict[i] for i in self.shaped_characters]
        protos = self.get_protos_idx(trsps, trchs, idx)
        plabels, sembs, tdicts = self.get_plabel_and_dict(trsps, trchs)
        return protos, sembs, plabels, tdicts
