import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F
from network.models.RPNLayer import RPNLayer
from util.util import *
from util import consts

class EntityMentionDetectionLayer(nn.Module):
    def __init__(self, hyps, entity_reprensentation_len):
        super(EntityMentionDetectionLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        # Move to right device
        self.cuda()
        self.entity_reprensentation_len = entity_reprensentation_len
        self.max_candidate_num = hyps["entity_candidate_num"]
        if self.hyperparams["EMD_enable"]:
            self.entity_detection_layer = RPNLayer(hyps, input_size=self.entity_reprensentation_len,
                                                   anchor_num=consts.ENTITY_ANCHOR_NUM,
                                                   sample_num=hyps["entity_sampled_number"],
                                                   weight=torch.FloatTensor([1., hyps["entity_det_weight"]]))
        
        self.dropout = nn.Dropout(hyps["dropout"])

    def no_forward(self, seq_mask):
        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_label = torch.zeros([BATCH_SIZE, SEQ_LEN, consts.ENTITY_ANCHOR_NUM], dtype=torch.int64).cuda()
        return zero_loss, zero_label, zero_loss, zero_label

    def forward(self, seq_mask, word_representation, cnn_representation, entity_anchor_loc,
                entity_anchor_labels, entity_anchor_type):
        '''

        '''
        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_label = torch.zeros([BATCH_SIZE, SEQ_LEN, consts.ENTITY_ANCHOR_NUM], dtype=torch.int64).cuda()

        reg_entity_representation = word_representation
        if self.hyperparams["EMD_enable"]:
            # (batch_size, seq_len, out) -》( batch, seqlen,  anchor_num, 2)
            loss_emd, detect_label, candidates_idx = self.entity_detection_layer(
                reg_entity_representation, entity_anchor_labels)
        else:
            loss_emd, detect_label = zero_loss, zero_label
            candidates_idx = torch.nonzero(entity_anchor_labels == 1)

        if self.hyperparams["EMD_cls_enable"]:
            loss_emd_cls, cls_label, entity_candidates_repr, entity_candidates_label, entity_candidates_predict, \
            entity_candidates_num, entity_candidates_len, entity_candidates_mask, entity_candidates_loc = \
                self.entity_classification_layer(word_mask=seq_mask,
                                                  word_repr=reg_entity_representation,
                                                  candidates_idx=candidates_idx,
                                                  anchor_loc=entity_anchor_loc,
                                                  anchor_label=entity_anchor_labels,
                                                  anchor_cls=entity_anchor_type)
        else:
            loss_emd_cls, cls_label = zero_loss, zero_label
            #entity_candidates_repr, entity_candidates_label, entity_candidates_num, entity_candidates_len, entity_candidates_mask, \
            #    entity_candidates_loc = self.entity_candidates_layer(reg_entity_representation, candidates_idx,
            #                                                             entity_anchor_loc, entity_anchor_type,
            #                                                             batch_candidate_num=self.max_candidate_num,
            #                                                             nonzero=True)

        return loss_emd, detect_label, loss_emd_cls, cls_label  # , reg_entity_representation, \
            #entity_candidates_repr, entity_candidates_label, entity_candidates_num, entity_candidates_len, \
            #entity_candidates_mask, entity_candidates_loc


    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))