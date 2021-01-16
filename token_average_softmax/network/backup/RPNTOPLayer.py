import copy
import numpy as np
import torch
from torch import nn
from util.consts import CUTOFF, VISIABLE
from util.util import *
from util.tensorSetOp import indexSetOp3d

class RPNLayer(nn.Module):
    def __init__(self, hyps, input_size, anchor_num, class_num, sample_num, candidate_num=0, weight=None):
        super(RPNLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.anchor_num = anchor_num
        self.class_num = class_num
        self.sample_num = sample_num
        self.candidate_num = candidate_num

        self.linear = torch.nn.Linear(in_features=input_size, out_features=anchor_num * class_num, bias=True)
        init_linear_(self.linear)
        #self.sizeof("rpn linear", self.linear.weight)
        #self.loss = nn.CrossEntropyLoss(weight=weight)
        self.loss = nn.MultiMarginLoss(p=1, margin=5, weight=weight)

    def forward(self, batch_input, anchor_labels):
        '''
        extracting event triggers
        anchor_labels: (batch, seqlen, 3)
        :param batch_input: FloatTensor, representation of the sentences in a batch, (batch_size, seq_len, dim)
                :param w_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)
        '''
        # [batch, seqlen, anchornum, dim]
        BATCH_SIZE, SEQ_LEN = batch_input.size()[:2]
        logits = self.linear(batch_input)  # batch, seqlen, anchor_num, class_num(2)
        #rpn_prob = F.softmax(logits.view([-1, self.anchor_num, 2]), dim=2)  # get the probabilty for each anchor
        batchfy_rpn_prob = logits.view(BATCH_SIZE, SEQ_LEN, self.anchor_num, self.class_num)  # batch, seqlen,  anchor_num, 2

        predict_label = torch.max(batchfy_rpn_prob, dim=3)[1]

        if self.training:
            positive_idx = torch.nonzero(anchor_labels == 1)
            logits_mask = torch.sigmoid(logits.clone())
            if positive_idx.size()[0] > 0:  # 直接调整logits的值使得golden一定最大，可以省掉合并过程
                logits_mask[positive_idx[:, 0], positive_idx[:, 1], 2 * positive_idx[:, 2] + 1] = 2.
            total_idx = self.get_topk_idx(logits_mask, BATCH_SIZE, SEQ_LEN, k=self.sample_num)
            #total_idx = indexSetOp3d(candidate_idx, positive_idx, operator="union")
            candidate_idx = total_idx
        else:
            total_idx = torch.nonzero(anchor_labels != -1)
            candidate_idx = self.get_topk_idx(logits, BATCH_SIZE, SEQ_LEN, k=self.candidate_num)
        sampled_rpn_label = anchor_labels[total_idx[:, 0], total_idx[:, 1], total_idx[:, 2]]  # sample_num
        sampled_rpn_prob = batchfy_rpn_prob[total_idx[:, 0], total_idx[:, 1], total_idx[:, 2]]  # sample_num x 2
        loss = self.loss(sampled_rpn_prob, sampled_rpn_label)
        candidate_label = predict_label[candidate_idx[:, 0], candidate_idx[:, 1], candidate_idx[:, 2]]
        return loss, predict_label, candidate_idx, candidate_label  #

    def get_topk_idx(self, logits, BATCH_SIZE, SEQ_LEN, k):
        max_k = max((1, k))
        _, batch_topk_pos = logits.view(BATCH_SIZE, SEQ_LEN * self.anchor_num, self.class_num).topk(max_k, 1, True,
                                                                                                    True)

        position_ = batch_topk_pos[:, :, 1].view(BATCH_SIZE * k, 1)  # top k of 1 class in each batch, [batch * k]
        seq_position = position_ // self.anchor_num
        anchor_position = position_ % self.anchor_num

        batch_matrix = np.array([range(BATCH_SIZE)] * k).transpose()
        batch_id = torch.LongTensor(batch_matrix).cuda().contiguous().view(BATCH_SIZE * k, 1)
        candidate_idx_ = torch.cat([batch_id, seq_position, anchor_position], dim=1)
        return candidate_idx_

    def sizeof(self, name, tensor):
        if not VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))


