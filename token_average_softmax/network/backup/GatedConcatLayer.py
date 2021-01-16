import copy

import torch
from torch import nn
from util.util import *
from util.consts import TRIGGER_ANCHOR_NUM, VISIABLE, CUTOFF

class GatedConcatLayer(nn.Module):
    def __init__(self, hyps, tensor1_len, tensor2_len, bn_func=nn.BatchNorm1d, seq_len=CUTOFF, bn_enable=True, project=False, gate_repr=True):
        super(GatedConcatLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.project = project
        self.gate_repr = gate_repr

        self.max_dim = max(tensor1_len, tensor2_len)
        if project:  # project both tensor_1 and 2 into same dim, then cat them
            reprensentation_len = 2 * self.max_dim
        else:
            reprensentation_len = tensor1_len + tensor2_len
        self.out_dim = reprensentation_len
        print("gate repr = {}".format(reprensentation_len))

        if project:
            self.tensor1_ln = nn.Linear(tensor1_len, self.max_dim, bias=True)
            init_linear_(self.tensor1_ln)
            self.tensor1_bn = bn_func(seq_len) if bn_enable else None
            self.tensor2_ln = nn.Linear(tensor2_len, self.max_dim, bias=True)
            init_linear_(self.tensor2_ln)
            self.tensor2_bn = bn_func(seq_len) if bn_enable else None

        if gate_repr:
            self.reg_gate_layer = nn.Linear(reprensentation_len, self.max_dim, bias=True)
            init_linear_(self.reg_gate_layer)
            self.reg_bn = bn_func(seq_len) if bn_enable else None
            self.out_dim = self.max_dim

        self.dropout = nn.Dropout(hyps["dropout"])
        # Move to right device

    def forward(self, tensor1, tensor2):
        '''
        extracting event triggers
        :return:

        '''
        if self.project:  # concat two part through project op
            tensor1 = self.tensor1_ln(tensor1)
            tensor2 = self.tensor2_ln(tensor2)

        cat_tensor = torch.cat([tensor1, tensor2], dim=-1)
        if self.gate_repr:
            #reg_gate_value = torch.sigmoid(self.reg_bn(self.reg_gate_layer(cat_tensor)))
            if self.reg_bn is not None:
                reg_gate_value = torch.sigmoid(self.reg_bn(self.reg_gate_layer(cat_tensor)))
            else:
                reg_gate_value = torch.sigmoid(self.reg_gate_layer(cat_tensor))
            reg_cat_tensor = reg_gate_value * tensor1 + (1 - reg_gate_value) * tensor2
        else:
            reg_cat_tensor = cat_tensor

        #reg_cat_tensor = self.dropout(reg_cat_tensor)  # if self.training else cat_tensor
        return reg_cat_tensor


    def sizeof(self, name, tensor):
        if not VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))

    def pad_to_max(self, origin_tensor):
        batch_size, seq_len, ori_dim = origin_tensor.size()[:3]
        padding = torch.zeros([batch_size, seq_len, self.max_dim]).cuda(self.device)
        padding[:, :, :ori_dim] = origin_tensor
        return padding