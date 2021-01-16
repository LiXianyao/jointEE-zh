import copy

import torch
from torch import nn
from util.util import *
from util.consts import TRIGGER_ANCHOR_NUM, VISIABLE, CUTOFF

class GateLayer(nn.Module):
    def __init__(self, hyps, tensor1_len, tensor2_len, bn_enable=True):
        super(GateLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.bn_enable = bn_enable
        self.cuda()

        reprensentation_len = tensor1_len + tensor2_len
        self.out_dim = tensor1_len
        #print("gate add repr = {}".format(reprensentation_len))

        self.reg_gate_layer = nn.Linear(reprensentation_len, self.out_dim, bias=True)
        init_linear_(self.reg_gate_layer)
        self.reg_bn = nn.BatchNorm1d(CUTOFF) if bn_enable else None

        self.dropout = nn.Dropout(hyps["dropout"])
        # Move to right device

    def forward(self, tensor1, tensor2):
        '''
        extracting event triggers
        :return:
        '''

        cat_tensor = torch.cat([tensor1, tensor2], dim=-1)
        if self.bn_enable:
            reg_gate_value = torch.sigmoid(self.reg_bn(self.reg_gate_layer(cat_tensor)))
        else:
            reg_gate_value = torch.sigmoid(self.reg_gate_layer(cat_tensor))
        reg_tensor = reg_gate_value * tensor1

        #reg_tensor = self.dropout(reg_tensor)  # if self.training else cat_tensor
        return reg_tensor


    def sizeof(self, name, tensor):
        if not VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))