import torch
from neko_sdk.ocr_modules.prototypers.neko_abstract_prototyper import neko_abstract_prototyper
import numpy as np


# produces prototypes from standard visual clue
# round robin to make the cache as updated as possible.
class neko_abstract_visual_center_prototyper(neko_abstract_prototyper):

    def pick(self, text):
        pass

    def get_label_set(self, label):
        return None, None

    def debug_show_protos(self, text, clabel, valid, nidx, cvProto):
        vProto = cvProto[0][:, 0, :, :]
        rmap = np.array(list(range(nidx.shape[0])))
        for i in range(nidx.shape[0]):
            rmap[nidx[i]] = i - 2
        # import cv2;
        # for i in range(len(text)):
        #     ims=[];
        #     print(text[i]);
        #     clab=clabel[i];
        #     for i in clab:
        #         if(rmap[i]>=0):
        #             ims.append(vproto[rmap[i]]);
        #     im=np.zeros([32*len(ims),32])
        #     for iid in range(len(ims)):
        #         im[iid*32:iid*32+32]=ims[iid];
        #     cv2.imshow(text[i],im);
        #     cv2.waitKey(0);
        #     pass;
        #     pass;

    def encode_compressed(self, text, batch_max_length):
        label, length = self.encode(text, batch_max_length)
        labels_in_episode, nidx = self.get_label_set(label)
        clabel, cproto, cvproto = self.label_proto_compress(label, labels_in_episode, nidx, None)
        self.debug_show_protos(text, clabel, labels_in_episode, nidx, cvproto)
        return label, length, labels_in_episode, nidx, clabel, cproto, cvproto

    def proto_compress(self, labels_in_task):
        norm_valid = labels_in_task[labels_in_task >= self.sp_cnt] - self.sp_cnt
        # if(norm_valid.shape[0]!=254):
        #     pass;
        norm_chars = torch.stack([self.norm_protos[i][0] for i in norm_valid], 0).repeat([1, 3, 1, 1]).contiguous()
        norm_weights = self.proto_engine(norm_chars.to(self.dev_ind.device))
        spproto = self.sp_protos / self.sp_protos.norm(dim=-1, keepdim=True)
        compact_prototype = torch.cat([spproto, norm_weights], dim=0)
        # print(compact_prototype.norm(dim=1));
        return compact_prototype, (norm_chars, norm_weights)

    def encode_text_compressed(self, text, nidx, labels_in_task, batch_max_length):
        label, length = self.encode(text, batch_max_length)
        clabel = self.label_compress(label, labels_in_task, nidx)
        return clabel, length

    def label_compress(self, label, labels_in_task, nidx):
        compact_ids = torch.zeros_like(label) + self.label_dict["[UNK]"]  # fill with unk
        for i in range(len(labels_in_task)):
            # print("here");
            compact_ids[label == labels_in_task[i]] = nidx[i]
        return compact_ids

    def label_proto_compress(self, label, labels_in_task, nidx, prototypes):
        compact_prototype, (norm_chars, norm_weights) = self.proto_compress(labels_in_task)
        compact_ids = self.label_compress(label, labels_in_task, nidx)
        # print(compact_prototype.norm(dim=1))
        return compact_ids, compact_prototype, (norm_chars, norm_weights)

    def label_decompress(self, compressed_ids, valid, index):
        original_id = valid[compressed_ids]
        # original_id=ids.clone();
        # for i in range(len(valid)):
        #     original_id[original_id == index[i]] =valid[i];
        return original_id
