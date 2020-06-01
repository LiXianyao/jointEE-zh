import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F
from util.util import *
from util import consts

class EventDetectionLayer(nn.Module):
    def __init__(self, hyps, trigger_repr_len):
        super(EventDetectionLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()

        self.detection_input_dim = trigger_repr_len
        self.classification_input_dim = trigger_repr_len

        self.out_dim = self.classification_input_dim
        self.dropout = nn.Dropout(hyps["dropout"])

    def no_forward(self, seq_mask):
        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_label = torch.zeros([BATCH_SIZE, SEQ_LEN, consts.TRIGGER_ANCHOR_NUM], dtype=torch.int64).cuda()
        return zero_loss, zero_label

    def forward(self, seq_mask, cnn_representation, word_representation, sentence_representation, trigger_anchor_loc,
                trigger_anchor_labels, trigger_anchor_type, trigger_bio_label,
                entity_candidates_repr, entity_candidates_mask, entity_candidates_len, entity_candidates_loc):

        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_label = torch.zeros([BATCH_SIZE, SEQ_LEN, consts.TRIGGER_ANCHOR_NUM], dtype=torch.int64).cuda()

        if self.hyperparams["gate_repr"]:
            print("use Bert-Feature GateCat CNN") if consts.ONECHANCE else None
            reg_trigger_representation = self.reg_gate_cnn_repr(word_representation, cnn_representation)
            if self.hyperparams["cat_hn"]:
                reg_trigger_representation = self.linear_hn(torch.cat([reg_trigger_representation, sentence_representation], dim=-1))
                reg_trigger_representation = self.dropout(torch.relu(self.bn_h(reg_trigger_representation)))
        else:
            print("use Bert-Feature cat CNN ") if consts.ONECHANCE else None
            word_representation = torch.cat([word_representation, sentence_representation], dim=-1) \
                if self.hyperparams["cat_hn"] else word_representation
            reg_trigger_representation = self.dropout(torch.cat([word_representation, cnn_representation], dim=-1))
        print("reg_trigger_repr_size is {}".format(reg_trigger_representation.size())) if consts.ONECHANCE else None

        if self.hyperparams["ED_enable"]:
            #(batch_size, seq_len, out) -》( batch, seqlen,  anchor_num, 2)
            loss_ed, detect_label, candidates_idx = self.trigger_detection_layer(
                reg_trigger_representation, trigger_anchor_labels)
            if self.hyperparams["ed_cls_mode"] == "roi":
                loss_ed_cls, cls_label, trigger_candidates_repr, trigger_candidates_label, trigger_candidates_predict, \
                    trigger_candidates_num, trigger_candidates_len, trigger_candidates_mask, trigger_candidates_loc = \
                    self.trigger_classification_layer(word_mask=seq_mask,
                                                      word_repr=reg_trigger_representation,
                                                      candidates_idx=candidates_idx,
                                                      anchor_loc=trigger_anchor_loc,
                                                      anchor_label=trigger_anchor_labels,
                                                      anchor_cls=trigger_anchor_type)
                trigger_candidates_weight = [{} for _ in range(BATCH_SIZE)]
                key_candidate_att_repr = None
            else:  # self.hyperparams["ed_cls_mode"] == "roi_key":
                loss_ed_cls, cls_label, trigger_candidates_repr, trigger_candidates_label, trigger_candidates_predict, \
                    trigger_candidates_num, trigger_candidates_len, trigger_candidates_mask, trigger_candidates_loc\
                    , trigger_candidates_weight, key_candidate_att_repr = \
                    self.trigger_classification_layer.forward(
                        word_mask=seq_mask, word_repr=reg_trigger_representation,
                        candidates_idx=candidates_idx, anchor_loc=trigger_anchor_loc,
                        anchor_cls=trigger_anchor_type,
                        key_candidates=entity_candidates_repr,
                        key_candidate_mask=entity_candidates_mask,
                        key_candidate_len=entity_candidates_len,
                        key_candidate_loc=entity_candidates_loc)
        else:
            loss_ed, detect_label, candidates_idx = zero_loss, zero_label, torch.nonzero(trigger_anchor_labels != -1)
            loss_ed_cls, cls_label = zero_loss, zero_label
            trigger_candidates_repr, trigger_candidates_label, trigger_candidates_num, trigger_candidates_len, \
                trigger_candidates_mask, trigger_candidates_loc = \
                self.trigger_candidates_layer(reg_trigger_representation, candidates_idx,
                                              trigger_anchor_loc, trigger_anchor_type,
                                              batch_candidate_num=self.hyperparams["trigger_candidate_num"],
                                              nonzero=False)
            trigger_candidates_predict = torch.zeros(trigger_candidates_label.size())
            trigger_candidates_weight = [{} for _ in range(BATCH_SIZE)]
            key_candidate_att_repr = None

        return loss_ed, detect_label, loss_ed_cls, cls_label, reg_trigger_representation, trigger_candidates_repr, \
            trigger_candidates_label, trigger_candidates_predict, trigger_candidates_num, trigger_candidates_len, \
            trigger_candidates_mask, trigger_candidates_loc, trigger_candidates_weight, key_candidate_att_repr


    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))

    def pad_to_max(self, origin_tensor):
        batch_size, seq_len, ori_dim = origin_tensor.size()[:3]
        padding = torch.zeros([batch_size, seq_len, self.max_dim]).cuda()
        padding[:, :, :ori_dim] = origin_tensor
        return padding