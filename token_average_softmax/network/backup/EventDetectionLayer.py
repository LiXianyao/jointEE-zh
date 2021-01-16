import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F
from util.util import *
from util import consts
#from .RPNLayer import RPNLayer
from .RPNTOPLayer import RPNLayer
from .RoiLayer2 import RoiLayer
from .GatedConcatLayer import GatedConcatLayer
from .GateLayer import GateLayer
from util.Anchor_enumerate import get_anchor_repr

class EventDetectionLayer(nn.Module):
    def __init__(self, hyps, trigger_repr_len):
        super(EventDetectionLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()

        self.trigger_reprensentation_dim = trigger_repr_len

        if self.hyperparams["gate_repr"] == False:
            self.trigger_reprensentation_dim += trigger_repr_len
        else:
            if self.hyperparams["ed_gate"] == "GC":
                print("trigger use gate Cat Layer")
                self.reg_gate_cnn_repr = GatedConcatLayer(hyps, tensor1_len=trigger_repr_len,
                                                   tensor2_len=trigger_repr_len,
                                                   bn_enable=True)
            else:
                print("trigger use gate Layer")
                self.reg_gate_cnn_repr = GateLayer(hyps, tensor1_len=trigger_repr_len,
                                                          tensor2_len=trigger_repr_len,
                                                          bn_enable=True)

        if self.hyperparams["ED_enable"]:
            self.trigger_detection_layer = RPNLayer(hyps, input_size=self.trigger_reprensentation_dim,
                                                    anchor_num=consts.TRIGGER_ANCHOR_NUM,
                                                    sample_num=hyps["trigger_sampled_number"],
                                                    class_num=2,
                                                    candidate_num=hyps["trigger_candidate_num"],
                                                    weight=torch.FloatTensor([1./hyps["trigger_det_weight"], 1.]))
            if self.hyperparams["ed_cls_mode"] == "roi":
                self.trigger_classification_layer = RoiLayer(hyps, input_size=self.trigger_reprensentation_dim,
                                                             anchor_num=consts.TRIGGER_ANCHOR_NUM,
                                                             class_num=hyps["trigger_label_weight"].size()[0],
                                                             weight=hyps["trigger_label_weight"],
                                                             use_att=hyps["ed_use_att"],
                                                             max_candidate_num=max(hyps["trigger_candidate_num"], hyps["trigger_sampled_number"]),
                                                             key_candidate_num=max(hyps["entity_candidate_num"], hyps["entity_sampled_number"])
                                                             )
                self.out_dim = self.trigger_reprensentation_dim
        self.out_dim = self.trigger_reprensentation_dim
        self.dropout = nn.Dropout(hyps["dropout"])

    def no_forward(self, seq_mask):
        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_label = torch.zeros([BATCH_SIZE, SEQ_LEN, consts.TRIGGER_ANCHOR_NUM], dtype=torch.int64).cuda()
        return zero_loss, zero_label

    def forward(self, seq_mask, cnn_representation, word_representation, trigger_anchor_loc,
                trigger_anchor_labels, trigger_anchor_type #):  #, sentence_representation,, trigger_bio_label
                ,entity_candidates_repr, entity_candidates_mask, entity_candidates_len, entity_candidates_loc):

        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_label = torch.zeros([BATCH_SIZE, SEQ_LEN, consts.TRIGGER_ANCHOR_NUM], dtype=torch.int64).cuda()

        if self.hyperparams["gate_repr"]:
            reg_trigger_representation = self.reg_gate_cnn_repr(word_representation, cnn_representation)
        else:
            #word_representation = torch.cat([word_representation, sentence_representation], dim=-1) \
            #    if self.hyperparams["cat_hn"] else word_representation
            reg_trigger_representation = self.dropout(torch.cat([word_representation, cnn_representation], dim=-1))
        print("reg_trigger_repr_size is {}".format(reg_trigger_representation.size())) if consts.ONECHANCE else None

        if self.hyperparams["ED_enable"]:
            """construct representation for classification"""
            trigger_anchor_representation = get_anchor_repr(hyps=self.hyperparams, anchor_const=consts.TRIGGER_ANCHOR_NUM,
                                                           BATCH_SIZE=BATCH_SIZE, SEQ_LEN=SEQ_LEN,
                                                           repr_dim=self.trigger_reprensentation_dim,
                                                           representation=reg_trigger_representation)

            #(batch_size, seq_len, out) -》( batch, seqlen,  anchor_num, 2)
            loss_ed, detect_label, candidates_idx, candidate_label = self.trigger_detection_layer.forward(
                reg_trigger_representation, trigger_anchor_labels)
            if self.hyperparams["ed_cls_mode"] == "roi":
                loss_ed_cls, cls_label, trigger_candidates_repr, trigger_candidates_label, trigger_candidates_predict, \
                    trigger_candidates_num, trigger_candidates_len, trigger_candidates_mask, trigger_candidates_loc, \
                    trigger_candidates_weight = \
                    self.trigger_classification_layer(word_mask=seq_mask,
                                                      word_repr=trigger_anchor_representation,
                                                      candidates_idx=candidates_idx,
                                                      candidate_label=candidate_label,
                                                      anchor_loc=trigger_anchor_loc,
                                                      anchor_label=trigger_anchor_labels,
                                                      anchor_cls=trigger_anchor_type,
                                                      key_candidates=entity_candidates_repr,
                                                      key_candidate_mask=entity_candidates_mask,
                                                      key_candidate_len=entity_candidates_len,
                                                      key_candidate_loc=entity_candidates_loc,
                                                      batch_candidate_num=self.hyperparams["trigger_candidate_num"]
                                                      if not self.training else self.hyperparams["trigger_sampled_number"])
                key_candidate_att_repr = None
        else:
            loss_ed, detect_label, candidates_idx = zero_loss, zero_label, torch.nonzero(trigger_anchor_labels != -1)
            loss_ed_cls, cls_label = zero_loss, zero_label
            trigger_candidates_repr, trigger_candidates_label, trigger_candidates_num, trigger_candidates_len, \
                trigger_candidates_mask, trigger_candidates_loc = [None] * 6 # \
                #self.trigger_candidates_layer(reg_trigger_representation, candidates_idx,
                #                              trigger_anchor_loc, trigger_anchor_type,
                #                              batch_candidate_num=self.hyperparams["trigger_candidate_num"],
                #                              nonzero=False)
            trigger_candidates_predict = None #torch.zeros(trigger_candidates_label.size())
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