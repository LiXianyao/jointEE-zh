import copy

import torch
from torch import nn
from torch.nn import functional as F
#from .RPNLayer import RPNLayer
from .RPNTOPLayer import RPNLayer
from util.util import *
from util import consts
from .GateLayer import GateLayer
from .GatedConcatLayer import GatedConcatLayer
from .RoiLayer2 import RoiLayer
from util.Anchor_enumerate import get_anchor_repr


class EntityMentionDetectionLayer(nn.Module):
    def __init__(self, hyps, entity_reprensentation_dim):
        super(EntityMentionDetectionLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        # Move to right device
        self.cuda()
        self.entity_reprensentation_dim = entity_reprensentation_dim
        self.max_candidate_num = hyps["entity_candidate_num"]

        if self.hyperparams["gate_repr"] == False:
            self.entity_reprensentation_dim += entity_reprensentation_dim
        else:
            if self.hyperparams["emd_gate"] == "GC":
                print("entity use gate Cat Layer")
                self.reg_gate_cnn_repr = GatedConcatLayer(hyps, tensor1_len=entity_reprensentation_dim,
                                                          tensor2_len=entity_reprensentation_dim,
                                                          bn_enable=True)
                self.reg_gate_anchor_repr = GatedConcatLayer(hyps, tensor1_len=entity_reprensentation_dim,
                                                          tensor2_len=entity_reprensentation_dim,
                                                          bn_func=nn.BatchNorm2d,
                                                          bn_enable=True)
            else:
                print("entity use gate Layer")
                self.reg_gate_cnn_repr = GateLayer(hyps, tensor1_len=entity_reprensentation_dim,
                                                          tensor2_len=entity_reprensentation_dim,
                                                          bn_enable=True)

        if self.hyperparams["EMD_enable"]:
            self.entity_detection_layer = RPNLayer(hyps, input_size=self.entity_reprensentation_dim,
                                                   anchor_num=consts.ENTITY_ANCHOR_NUM,
                                                   class_num=2,
                                                   sample_num=hyps["entity_sampled_number"],
                                                   candidate_num=hyps["entity_candidate_num"],
                                                   weight=torch.FloatTensor([1. / hyps["entity_det_weight"], 1.]))
            if self.hyperparams["EMD_cls_enable"]:
                self.entity_classification_layer = RoiLayer(hyps, input_size=self.entity_reprensentation_dim,
                                                            anchor_num=consts.ENTITY_ANCHOR_NUM,
                                                            class_num=hyps["entity_label_weight"].size()[0],
                                                            weight=hyps["entity_label_weight"],
                                                            use_att=hyps["emd_use_att"],
                                                            max_candidate_num=max(hyps["entity_candidate_num"],
                                                                                   hyps["entity_sampled_number"])
                                                            )
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

        if self.hyperparams["gate_repr"]:
            print("use Bert-Feature GateCat CNN AT entity") if consts.ONECHANCE else None
            reg_entity_representation = self.reg_gate_cnn_repr(word_representation, cnn_representation)
        else:
            print("use Bert-Feature cat CNN AT entity") if consts.ONECHANCE else None
            reg_entity_representation = torch.cat([word_representation, cnn_representation], dim=-1)

        """construct representation for classification"""
        entity_anchor_representation = get_anchor_repr(hyps=self.hyperparams, anchor_const=consts.ENTITY_ANCHOR_NUM,
                                                       BATCH_SIZE=BATCH_SIZE, SEQ_LEN=SEQ_LEN,
                                                       repr_dim=self.entity_reprensentation_dim,
                                                       representation=reg_entity_representation)

        if self.hyperparams["EMD_enable"]:
            # (batch_size, seq_len, out) -》( batch, seqlen,  anchor_num, 2)
            loss_emd, detect_label, candidates_idx, candidate_label = self.entity_detection_layer.forward(
                reg_entity_representation, entity_anchor_labels)
        else:
            loss_emd, detect_label = zero_loss, zero_label
            candidates_idx = torch.nonzero(entity_anchor_labels == 1)

        if self.hyperparams["EMD_cls_enable"]:
            loss_emd_cls, cls_label, entity_candidates_repr, entity_candidates_label, entity_candidates_predict, \
                entity_candidates_num, entity_candidates_len, entity_candidates_mask, entity_candidates_loc, \
                entity_candidates_weight = \
                self.entity_classification_layer.forward(word_mask=seq_mask,
                                                         word_repr=entity_anchor_representation,
                                                         candidates_idx=candidates_idx,
                                                         candidate_label=candidate_label,
                                                         anchor_loc=entity_anchor_loc,
                                                         anchor_label=entity_anchor_labels,
                                                         anchor_cls=entity_anchor_type,
                                                         batch_candidate_num=self.hyperparams["entity_candidate_num"]
                                                         if not self.training else self.hyperparams["entity_sampled_number"])
        else:
            loss_emd_cls, cls_label = zero_loss, zero_label
            entity_candidates_repr, entity_candidates_label, entity_candidates_num, entity_candidates_len, entity_candidates_mask, \
                entity_candidates_loc = [None] * 6 #self.entity_candidates_layer(entity_anchor_representation, candidates_idx,
                                                    #                 entity_anchor_loc, entity_anchor_type,
                                                     ##\                batch_candidate_num=self.max_candidate_num,
                                                        #             nonzero=True)
            entity_candidates_predict = None

        return loss_emd, detect_label, loss_emd_cls, cls_label, entity_anchor_representation,\
            entity_candidates_repr, entity_candidates_label, entity_candidates_predict, entity_candidates_num, \
            entity_candidates_len, \
            entity_candidates_mask, entity_candidates_loc


    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))