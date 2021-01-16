import copy

import torch
from torch import nn
from util import consts
from util.util import *

class AttentionLayer(nn.Module):
    def __init__(self, hyps, input_size, max_candidate_num, key_candidate_num, query_mask=False):
        super(AttentionLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.max_candidate_num = max_candidate_num

        self.forget_gate_linear0 = nn.Linear(in_features=input_size, out_features=input_size, bias=False)
        init_linear_(self.forget_gate_linear0)
        self.forget_gate_linear1 = nn.Linear(in_features=input_size, out_features=input_size, bias=False)
        init_linear_(self.forget_gate_linear1)
        self.bn0 = torch.nn.BatchNorm1d(max_candidate_num)
        self.bn1 = torch.nn.BatchNorm1d(key_candidate_num)

        if hyps["att_type"] == 'softmax':
            print("softmax roi with key")
            self.linear_v = nn.Linear(in_features=input_size, out_features=1, bias=False)
            init_linear_(self.linear_v)
            self.bnv = torch.nn.BatchNorm1d(max_candidate_num)
        else:
            print("gate roi with key")

        if hyps["att_activate"] == 'relu':
            self.activate_func = torch.relu
        elif hyps["att_activate"] == 'sigmoid':
            self.activate_func = torch.sigmoid
        else:
            self.activate_func = torch.tanh

        self.query_mask = query_mask
        if query_mask:
            print("query self att weight is {}".format(hyps["query_weight"]))
            self.attention_mask = torch.ones([max_candidate_num, key_candidate_num]).cuda()
            for i in range(max_candidate_num):
                self.attention_mask[i, i] = hyps["query_weight"]
            self.attention_mask = self.attention_mask.unsqueeze(-1)
        self.dropout = nn.Dropout(hyps["dropout"])

    def gated_attention(self, batch_candidates_repr, candidate_mask,
                        key_candidates_repr, key_mask, key_len):
        BATCH_SIZE, CANDIDATE_NUM, REPR_DIM = batch_candidates_repr.size()[:3]
        KEY_NUM = key_candidates_repr.size()[1]

        # attention-based context information
        # batch x max_seq x embed --> batch x max_seq x max_seq x embed
        forget_gate0_linear = self.forget_gate_linear0(batch_candidates_repr)
        forget_gate1_linear = self.forget_gate_linear1(key_candidates_repr)
        forget_gate0_linear = self.bn0(forget_gate0_linear)
        forget_gate1_linear = self.bn1(forget_gate1_linear)

        # take advantage of the broadcast to calculate the attention weight for CANDIDATE_NUM * CANDIDATE_NUM inputs
        input0 = forget_gate0_linear.view(BATCH_SIZE, CANDIDATE_NUM, 1, REPR_DIM)\
            .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)
        #input0 = torch.mul(input0, self.attention_mask)
        input1 = forget_gate1_linear.view(BATCH_SIZE, 1, KEY_NUM, REPR_DIM)\
                .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)
        if self.query_mask:
            input1 = torch.mul(input1, self.attention_mask)

        attention_weight = torch.mul(torch.sigmoid(input1), torch.sigmoid(input0))
        #attention_weight = torch.sigmoid(sigmoid_input)  # [BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM]

        input_epanded = key_candidates_repr.view(BATCH_SIZE, 1, KEY_NUM, REPR_DIM)\
            .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)   # 由每个KEY的权重 * KEY向量的值 求和 得到 QUERY的新表示
        attention_res = torch.mul(input_epanded, attention_weight)

        # mask the addition padding into zeros
        mask_expended = candidate_mask.view(BATCH_SIZE, CANDIDATE_NUM, 1) \
            .mul(key_mask.view(BATCH_SIZE, 1, KEY_NUM))   # calculate masked matrix among candidates in sen
        mask_expended = mask_expended.view(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, 1) \
            .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)

        attention_res_masked = torch.mul(attention_res, mask_expended.float())  # [b, c, k, dim]
        attention_vector = torch.sum(attention_res_masked, 2)  # ri = sum(rij · xj)
        attention_vector = torch.div(attention_vector,
                                     key_len.view(BATCH_SIZE, 1, 1).expand(BATCH_SIZE, CANDIDATE_NUM, REPR_DIM).float()
                                     )  # 虽然包上trigger实际上是 len + 1个
        output_res = self.activate_func(attention_vector)
        print("tanh and no bnatt ") if consts.ONECHANCE else None
        #output_res = attention_vector
        return output_res, attention_res_masked, torch.mean(attention_weight, dim=-1)

    def softmax_attention(self, batch_candidates_repr, candidate_mask,
                        key_candidates_repr, key_mask):
        BATCH_SIZE, CANDIDATE_NUM, REPR_DIM = batch_candidates_repr.size()[:3]
        KEY_NUM = key_candidates_repr.size()[1]

        # attention-based context information
        # batch x max_seq x embed --> batch x max_seq x max_seq x embed
        forget_gate0_linear = self.forget_gate_linear0(batch_candidates_repr)
        forget_gate1_linear = self.forget_gate_linear1(key_candidates_repr)
        forget_gate0_linear = self.bn0(forget_gate0_linear)
        forget_gate1_linear = self.bn1(forget_gate1_linear)

        # take advantage of the broadcast to calculate the attention weight for CANDIDATE_NUM * CANDIDATE_NUM inputs
        tanh_input = forget_gate0_linear.view(BATCH_SIZE, CANDIDATE_NUM, 1, REPR_DIM)\
            .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM) + \
            forget_gate1_linear.view(BATCH_SIZE, 1, KEY_NUM, REPR_DIM)\
                .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)
        tanh_input = self.activate_func(tanh_input)

        score = self.linear_v(tanh_input).view(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM)  # [b, c, k]
        score = self.bnv(score)
        attention_weight = torch.softmax(score, dim=2).view(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, 1)  # [b, c, k]

        input_epanded = key_candidates_repr.view(BATCH_SIZE, 1, KEY_NUM, REPR_DIM)\
            .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)   # 由每个KEY的权重 * KEY向量的值 求和 得到 QUERY的新表示
        attention_res = torch.mul(attention_weight, input_epanded)

        # mask the addition padding into zeros
        mask_expended = candidate_mask.view(BATCH_SIZE, CANDIDATE_NUM, 1) \
            .mul(key_mask.view(BATCH_SIZE, 1, KEY_NUM))   # calculate masked matrix among candidates in sen
        mask_expended = mask_expended.view(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, 1) \
            .expand(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM, REPR_DIM)

        attention_res_masked = torch.mul(attention_res, mask_expended.float())
        attention_vector = torch.sum(attention_res_masked, 2)  # ri = sum(rij · xj)
        output_res = self.activate_func(attention_vector)
        #print("tanh and no bnatt ") if consts.ONECHANCE else None
        #output_res = attention_vector
        return output_res, attention_res_masked, attention_weight.view(BATCH_SIZE, CANDIDATE_NUM, KEY_NUM)

    def forward(self, query_candidates_repr, query_candidate_mask, key_candidates, key_candidate_mask, key_candidate_len):
        if self.hyperparams["att_type"] == 'softmax':
            candidates_att_repr, key_candidates_repr, attention_weight = \
                self.softmax_attention(query_candidates_repr,
                                       query_candidate_mask,
                                       key_candidates, key_candidate_mask)
        else:
            candidates_att_repr, key_candidates_repr, attention_weight = self.gated_attention(query_candidates_repr, query_candidate_mask,
                                                                            key_candidates, key_candidate_mask,
                                                                            key_candidate_len)

        return candidates_att_repr, key_candidates_repr, attention_weight

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))
