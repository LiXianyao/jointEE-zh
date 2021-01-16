import copy

import torch
from torch import nn
from torch.nn import functional as F
from util import consts
from util.util import *

class CandidateRepresentationLayer(nn.Module):
    def __init__(self, hyps, input_size, anchor_num):
        super(CandidateRepresentationLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.anchor_num = anchor_num
        self.dropout = nn.Dropout(p=hyps["inception_dropout"])

    def forward(self, word_repr, candidates_idx, anchor_loc, anchor_cls, batch_candidate_num=None,
                            max_candidate_num=None, nonzero=True):
        '''
        extracting event triggers
        anchor_labels: (batch, seqlen, 3)
        :param batch_input: FloatTensor, representation of the sentences in a batch, (batch_size, seq_len, dim)
                :param w_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)
        '''

        # get candidates : BATCH_SIZE, batch_candidate_num, anchor_num, REPR_DIM
        batch_candidates_repr, batch_candidates_label, batch_candidate_num, candidate_len, candidate_mask, \
            candidate_loc = self.get_candidates_repr(
                word_repr, candidates_idx, anchor_loc, anchor_cls, anchor_num=self.anchor_num,
                batch_candidate_num=batch_candidate_num, max_candidate_num=max_candidate_num, nonzero=nonzero)
        return batch_candidates_repr, batch_candidates_label, batch_candidate_num, candidate_len, candidate_mask, candidate_loc

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))

    def get_candidates_repr(self, word_repr, candidates_idx, anchor_loc, anchor_cls, anchor_num, batch_candidate_num=None,
                            max_candidate_num=None, nonzero=True):
        BATCH_SIZE, SEQ_LEN, anchor_num, REPR_DIM = word_repr.size()[:4]
        #print(word_repr.size())
        """calculate maximum candidates number for each sentence in the whole batch if it hasn't been provided"""
        batch_candidate_num = self.get_batch_candidate_num(BATCH_SIZE, candidates_idx) \
            if batch_candidate_num is None else batch_candidate_num
        if max_candidate_num is not None:
            batch_candidate_num = min(batch_candidate_num, max_candidate_num)

        # [batch * k, 3]
        #candidates_idx = candidates_idx.view(BATCH_SIZE, batch_candidate_num, 3)
        #word_repr = word_repr.view(BATCH_SIZE, SEQ_LEN, 1, REPR_DIM)
        batch_idx, wid, aid = candidates_idx[:, 0], candidates_idx[:, 1], candidates_idx[:, 2]
        batch_candidates_repr = word_repr[batch_idx, wid, aid, :].view(BATCH_SIZE, batch_candidate_num, REPR_DIM)

        batch_candidates_label = anchor_cls[batch_idx, wid, aid].view(BATCH_SIZE, batch_candidate_num)
        candidate_loc = anchor_loc[batch_idx, wid, aid]  # [b*k, 2]

        # remove the padding and out of boundary's candidate
        invalid_loc = torch.nonzero(candidate_loc[:, 0] == candidate_loc[:, 1])
        candidate_mask = torch.ones([BATCH_SIZE * batch_candidate_num], requires_grad=False, dtype=torch.float).cuda()
        candidate_mask[invalid_loc[:, 0]] = 0.
        candidate_mask = candidate_mask.view(BATCH_SIZE, batch_candidate_num)
        candidate_len = torch.sum(candidate_mask, dim=1)
        candidate_loc = candidate_loc.view(BATCH_SIZE, batch_candidate_num, 2)
        batch_candidates_repr = batch_candidates_repr * candidate_mask.unsqueeze(-1)

        for sid in range(BATCH_SIZE):  # in gate att, need to divide len, so change the 0 into 1
            if candidate_len[sid] == 0 and nonzero: candidate_len[sid] = 1

        return batch_candidates_repr, batch_candidates_label, batch_candidate_num, candidate_len, candidate_mask, candidate_loc

    def get_batch_candidate_num(self, BATCH_SIZE, candidates_idx):
        candidate_len = torch.zeros([BATCH_SIZE], dtype=torch.int64).cuda()
        if len(candidates_idx.size()) < 2 or candidates_idx.size()[1] < 0:
            print("zero candidates")
            return 1

        for idx in range(candidates_idx.size()[0]):
            batch_idx, wid, aid = candidates_idx[idx][:3]
            candidate_len[batch_idx] += 1
        return max(torch.max(candidate_len).item(), 1)

    def fill_candidates_prob(self, BATCH_SIZE, SEQ_LEN, logits, anchor_loc, candidates_idx, class_num, max_candidate_num):
        full_batch_prob = torch.zeros([BATCH_SIZE, SEQ_LEN, self.anchor_num, class_num]).cuda()
        batch_idx, wid, aid = candidates_idx[:, 0], candidates_idx[:, 1], candidates_idx[:, 2]  # [b*k, 3]

        logits = logits.view(BATCH_SIZE * max_candidate_num, class_num)
        full_batch_prob[batch_idx, wid, aid, :] = logits[:, :]

        return full_batch_prob

    def fill_candidates_weight(self, BATCH_SIZE, attention_weight, candidate_loc, candidates_predict,
                               candidates_label, key_candidate_loc):
        batch_candidates_weight = [{} for _ in range(BATCH_SIZE)]
        #if self.training: return batch_candidates_weight
        """
        {
            (sid, eid): [[(s,e), w], [(s,e), w] , ...]
        }
        """
        query_num = candidate_loc.size()[1]
        key_num = key_candidate_loc.size()[1]
        for idx in range(BATCH_SIZE):
            batch_candidates_weight[idx] = {}
            for qid in range(query_num):
                if candidate_loc[idx, qid, 0].item() == 0: continue  # 非句子内项目
                query_loc = tuple(candidate_loc[idx, qid, :2].tolist())
                query_predict = candidates_predict[idx, qid].item()
                query_label = candidates_label[idx, qid].item()
                batch_candidates_weight[idx][query_loc] = {"label": query_label, "predict": query_predict,
                                                           "t_correct": query_label == query_predict, "keys": []}
                for kid in range(key_num):
                    if key_candidate_loc[idx, kid, 0].item() == 0: continue
                    key_loc = tuple(key_candidate_loc[idx, kid,:2].tolist())
                    batch_candidates_weight[idx][query_loc]["keys"].append(
                        (key_loc, attention_weight[idx][qid][kid].item())
                    )
        return batch_candidates_weight