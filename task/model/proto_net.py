from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core import neko_nonsematical_prototype_core_basic


class OracleProtoNet(neko_nonsematical_prototype_core_basic):

    def sample_charset_by_text(self, text_batch):
        tmpBatch = []
        for _, one in enumerate(text_batch):
            tmpBatch.append(one)
        trsps, trchs = self.get_sampled_ids(tmpBatch)
        trchs = list(trchs)
        trsps = list(trsps)
        # get proto -> proto engine -> resNet18
        protos = self.get_protos(trsps, trchs)
        plabels, sembs, tdicts = self.get_plabel_and_dict(trsps, trchs)
        # this.debug(trchs,"meow");
        return protos, sembs, plabels, tdicts

    def transLabel(self, label):
        masterLabel = []
        for one in label:
            if one in self.label_dict:
                masterIndex = self.label_dict[one]
                masterLabel.append(self.character[masterIndex])
            else:
                masterLabel.append(one)
        return masterLabel
