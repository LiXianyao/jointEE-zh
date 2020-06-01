import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.consts import CUTOFF, VISIABLE
from util.util import *
from util.tensorSetOp import indexSetOp3d

class RPNLayer(nn.Module):
    def __init__(self, hyps, input_size, anchor_num, class_num=2, sample_num=1, weight=None):
        super(RPNLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.anchor_num = anchor_num
        self.class_num = class_num
        self.sample_num = hyps["batch"] * sample_num

        self.linear = torch.nn.Linear(in_features=input_size, out_features=anchor_num * class_num, bias=True)
        self.bn = torch.nn.BatchNorm1d(CUTOFF)
        init_linear_(self.linear)
        #self.sizeof("rpn linear", self.linear.weight)
        self.loss = nn.CrossEntropyLoss(weight=weight)
        #print("create RPN Layer with shape of {}, and cls weight {}".format(self.linear.weight.size(), weight)) if consts.VISIABLE else None

    def forward(self, batch_input, anchor_labels):
        '''
        extracting event triggers
        anchor_labels: (batch, seqlen, 3)
        :param batch_input: FloatTensor, representation of the sentences in a batch, (batch_size, seq_len, dim)
                :param w_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)
        '''

        BATCH_SIZE, SEQ_LEN = batch_input.size()[:2]
        logits = self.linear(batch_input)  # batch, seqlen, 2* anchor_num
        #rpn_prob = F.softmax(logits.view([-1, self.anchor_num, 2]), dim=2)  # get the probabilty for each anchor
        batchfy_rpn_prob = logits.view(BATCH_SIZE, SEQ_LEN, self.anchor_num, self.class_num)  # batch, seqlen,  anchor_num, 2

        predict_label = torch.max(batchfy_rpn_prob, dim=3)[1]
        postive_predict_idx = torch.nonzero(predict_label == 1)
        padding_idx = torch.nonzero(anchor_labels == -1)

        def remove_padding_idx_and_shuffle(input_idx, sample_num=None, shuffle=True):
            input_idx = indexSetOp3d(input_idx, padding_idx, operator="diff")
            input_idx = shuffle_idx(input_idx, sample_num) if shuffle else input_idx
            return input_idx

        def shuffle_idx(input_idx, sample_num):
            if input_idx.size()[0] > 0:
                sample_array = np.array(range(input_idx.size()[0]))
                np.random.shuffle(sample_array)
                if sample_num is not None and sample_num > 0:
                    input_idx = input_idx[torch.tensor(sample_array[:sample_num])]  # sample_num x 2
                else:
                    input_idx = input_idx[torch.tensor(sample_array)]  # sample_num x 2
            return input_idx

        if self.training:
            positive_idx = shuffle_idx(torch.nonzero(anchor_labels == 1), self.sample_num)  # ground_truth
            # index of anchor_labels where is positive，
            pos_num = positive_idx.size()[0]

            #sampled_neg_num = 2 * pos_num if pos_num > 0 else self.sample_num  06
            sampled_neg_num = self.sample_num * 2 - pos_num
            # index of predict anchor that is positive, [number x 3]
            wrong_predict_idx = indexSetOp3d(postive_predict_idx, positive_idx, operator="diff")
            negative_idx = torch.nonzero(anchor_labels == 0)

            wrong_predict_idx = indexSetOp3d(wrong_predict_idx, negative_idx, operator="inter")
            wrong_predict_num = min(wrong_predict_idx.size()[0], sampled_neg_num)
            wrong_predict_idx = shuffle_idx(wrong_predict_idx, wrong_predict_num)

            sampled_neg_num -= wrong_predict_num
            sampled_neg_idx = shuffle_idx(negative_idx, max(sampled_neg_num, 1))  # sample_num x 3

            total_idx = torch.cat([positive_idx, wrong_predict_idx, sampled_neg_idx], 0)
            #candidate_idx = torch.nonzero(anchor_labels != -1)
            #candidate_idx = total_idx  # remove_padding_idx_and_shuffle(total_idx, shuffle=False)
            candidate_idx = torch.cat([positive_idx, wrong_predict_idx], 0)
        else:
            postive_predict_idx = remove_padding_idx_and_shuffle(postive_predict_idx, shuffle=False)

            """when testing, assume it don't have any pre-knowing labels, just sample the predicted res
            sampled_neg_num = postive_predict_idx.size()[0] if postive_predict_idx.size()[0] else self.sample_num
            negative_idx = remove_padding_idx_and_shuffle(torch.nonzero(predict_label == 0), sampled_neg_num)"""

            total_idx = torch.nonzero(anchor_labels != -1)
            #total_idx = torch.cat([postive_predict_idx, negative_idx], 0)  # when testing ,only care about the predict 1
            candidate_idx = postive_predict_idx
            #candidate_idx = torch.nonzero(anchor_labels != -1)
        sampled_rpn_label = anchor_labels[total_idx[:, 0], total_idx[:, 1], total_idx[:, 2]]  # sample_num
        sampled_rpn_prob = batchfy_rpn_prob[total_idx[:, 0], total_idx[:, 1], total_idx[:, 2]]  # sample_num x 2
        loss = self.loss(sampled_rpn_prob, sampled_rpn_label)
        return loss, predict_label, candidate_idx  #

    def sizeof(self, name, tensor):
        if not VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))


