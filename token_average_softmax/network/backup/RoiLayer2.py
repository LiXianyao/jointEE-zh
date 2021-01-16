import copy

import torch
from torch import nn
from util import consts
from util.util import *
from .CandidateRepresentationLayer import CandidateRepresentationLayer
from .SoftmaxAttentionLayer import AttentionLayer
from .GatedConcatLayer import GatedConcatLayer

class RoiLayer(nn.Module):
    def __init__(self, hyps, input_size, anchor_num, max_candidate_num, key_candidate_num, class_num, weight=None, use_att=False):
        super(RoiLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.anchor_num = anchor_num
        self.class_num = class_num
        self.max_candidate_num = max_candidate_num
        self.use_att = use_att

        self.candidateRepresentationLayer = CandidateRepresentationLayer(hyps, input_size=input_size,
                                                                                 anchor_num=anchor_num)
        if self.use_att:
            self.self_att_layer = AttentionLayer(hyps, input_size=input_size, max_candidate_num=max_candidate_num,
                                                 key_candidate_num=key_candidate_num, query_mask=True)
            self.gate_att_repr = GatedConcatLayer(hyps, tensor1_len=input_size,
                                                  tensor2_len=input_size,
                                                  seq_len=max_candidate_num,
                                                  bn_enable=True)

        self.linear = torch.nn.Linear(in_features=input_size, out_features=class_num, bias=True)
        init_linear_(self.linear)
        #self.sizeof("rpn linear", self.linear.weight)
        if weight is not None:
            #self.loss = nn.CrossEntropyLoss(weight=weight)
            self.loss = nn.MultiMarginLoss(p=1, margin=5, weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss()
        #print("create Roi Layer with shape of {}, and cls weight {}".format(self.linear.weight.size(), weight)) if consts.VISIABLE else None
        self.dropout = nn.Dropout(hyps["dropout"])

    def get_candidates_repr(self, word_repr, candidates_idx, anchor_loc, anchor_cls):
        batch_candidates_repr, batch_candidates_label, batch_candidate_num, candidate_len, candidate_mask, \
            candidate_loc = self.candidateRepresentationLayer(word_repr, candidates_idx, anchor_loc, anchor_cls,
                                                batch_candidate_num=self.max_candidate_num, nonzero=True)
        return batch_candidates_repr, candidate_mask

    def forward(self, word_mask, word_repr, candidates_idx, candidate_label, anchor_loc, anchor_label, anchor_cls,
                batch_candidate_num,
                key_candidates, key_candidate_mask, key_candidate_len, key_candidate_loc):
        '''
        extracting event triggers
        anchor_labels: (batch, seqlen, 3)
        :param batch_input: FloatTensor, representation of the sentences in a batch, (batch_size, seq_len, dim)
                :param w_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)
        '''

        BATCH_SIZE, SEQ_LEN, ANCHOR_NUM, REPR_DIM = word_repr.size()[:4]
        # get candidates : BATCH_SIZE, batch_candidate_num, REPR_DIM
        batch_candidates_repr, batch_candidates_label, batch_candidate_num, candidate_len, candidate_mask, \
            candidate_loc = self.candidateRepresentationLayer.forward(word_repr, candidates_idx,
                                                                      candidate_label=candidate_label,
                                                                      anchor_loc=anchor_loc,
                                                                      anchor_cls=anchor_cls,
                                                                      batch_candidate_num=batch_candidate_num,
                                                                      nonzero=True)
        if self.use_att == 0:
            # then linear+ln, get the cls for each candidates
            print("ROI classification method is  Linear ") if consts.ONECHANCE else None
            batch_candidates_logits = self.linear(batch_candidates_repr)  # BATCH_SIZE, batch_candidate_num, self.class_num
            attention_weight = None
        else:
            print("ROI classification method is  Att ") if consts.ONECHANCE else None
            candidates_att_repr, key_candidates_att_repr, attention_weight = self.self_att_layer.forward(
                batch_candidates_repr, candidate_mask,
                key_candidates, key_candidate_mask,
                key_candidate_len)
            batch_candidates_repr = self.gate_att_repr(batch_candidates_repr, candidates_att_repr)
            batch_candidates_logits = self.linear(batch_candidates_repr)  # BATCH_SIZE, batch_candidate_num, self.class_num
        batch_candidates_prob = batch_candidates_logits.view([BATCH_SIZE * batch_candidate_num, self.class_num])
        batch_candidates_predict = torch.max(batch_candidates_logits, 2)[1]

        if attention_weight is not None:
            batch_candidates_weight = self.candidateRepresentationLayer.fill_candidates_weight\
                (BATCH_SIZE, attention_weight, candidate_loc, batch_candidates_predict, batch_candidates_label,
                 candidate_loc)
        else:
            batch_candidates_weight = [{} for _ in range(BATCH_SIZE)]

        cls_loss = self.loss(batch_candidates_prob, batch_candidates_label.view([BATCH_SIZE * batch_candidate_num]))  #

        # BATCH_SIZE, SEQ_LEN, self.anchor_num, self.class_num
        full_batch_prob = self.candidateRepresentationLayer.fill_candidates_prob\
            (BATCH_SIZE, SEQ_LEN, batch_candidates_logits, anchor_loc, candidates_idx, class_num=self.class_num,
             max_candidate_num=batch_candidate_num)

        predict_label = torch.max(full_batch_prob, 3)[1]
        return cls_loss, predict_label, batch_candidates_repr, batch_candidates_label, batch_candidates_predict,\
            batch_candidate_num, candidate_len, candidate_mask, candidate_loc, batch_candidates_weight

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))
