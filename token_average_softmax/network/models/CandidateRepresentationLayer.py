import copy

from network.models.EmbeddingLayer import *
import torch
from torch import nn
from torch.nn import functional as F
from util import consts
from util.util import *

class CandidateRepresentationLayer(nn.Module):
    def __init__(self, hyps, anchor_num, strategy='mean'):
        super(CandidateRepresentationLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.anchor_num = anchor_num
        self.strategy = strategy
        self.dropout = nn.Dropout(p=hyps["inception_dropout"])

    def forward(self, word_repr, candidates_idx, anchor_loc, anchor_cls):
        '''
        extracting event triggers
        anchor_labels: (batch, seqlen, 3)
        :param batch_input: FloatTensor, representation of the sentences in a batch, (batch_size, seq_len, dim)
                :param w_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)
        '''

        # get candidates : BATCH_SIZE, batch_candidate_num, anchor_num, REPR_DIM
        batch_candidates_repr, batch_candidates_label, batch_candidates_len, batch_candidates_loc = \
            self.get_candidates_repr(
                word_repr, candidates_idx, anchor_loc, anchor_cls, anchor_num=self.anchor_num, strategy=self.strategy)

        return batch_candidates_repr, batch_candidates_label, batch_candidates_len, batch_candidates_loc

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))

    def get_candidates_repr(self, word_repr, candidates_idx, anchor_loc, anchor_cls, anchor_num, strategy='mean'):
        BATCH_SIZE, SEQ_LEN, REPR_DIM = word_repr.size()[:3]

        """select the repr of the candidates into one tensor"""
        batch_candidates_repr = {bid: [] for bid in range(BATCH_SIZE)}
        batch_candidates_label = {bid: [] for bid in range(BATCH_SIZE)}
        batch_candidates_len = [0 for _ in range(BATCH_SIZE)]
        batch_candidates_loc = {bid: [] for bid in range(BATCH_SIZE)}

        for idx in range(candidates_idx.size()[0]):
            batch_idx, wid, aid = candidates_idx[idx][:3]
            #print(batch_idx, wid, aid)
            sid, eid = anchor_loc[batch_idx, wid, aid][:2]
            if sid == eid: continue

            anchor_len = eid - sid
            if word_repr[batch_idx, sid: eid, :].size()[0] < anchor_len:
                print("len mismatch", sid, eid, SEQ_LEN)
            bid = batch_idx.item()
            batch_candidates_loc[bid].append(anchor_loc[batch_idx, wid, aid])
            batch_candidates_label[bid].append(anchor_cls[batch_idx, wid, aid])
            batch_candidates_len[batch_idx] += 1
            if strategy == 'first':  # only take the first word of the anchor as it's repr
                candidate_repr = word_repr[batch_idx, sid, :]
            else:  # mean
                candidate_repr = word_repr[batch_idx, sid: eid, :].mean(dim=-2)
            batch_candidates_repr[bid].append(candidate_repr)

        #for sid in range(BATCH_SIZE):  # in gate att, need to divide len, so change the 0 into 1
        #    if batch_candidates_len[sid] == 0 and nonzero: batch_candidates_len[sid] = 1

        return batch_candidates_repr, batch_candidates_label, batch_candidates_len, batch_candidates_loc
