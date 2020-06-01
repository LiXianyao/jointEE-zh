import torch
from torch import nn
import numpy as np
from util import consts

def enum_anchor_repr(word_repr, len_idx, anchor_num, previous_anchor_idx, batch_idx, batch_size):
    if anchor_num == 1:
        anchor_idx = len_idx
    elif anchor_num % 2 == 1:  # 补前位
        front_padding = (anchor_num - 1) // 2  # 3时候，前面补1位，5时候补两位...
        z = torch.zeros_like(len_idx)
        z[front_padding:] = len_idx[: -front_padding]
        anchor_idx = torch.cat([z, previous_anchor_idx], dim=-1)
    else:
        tail_padding = anchor_num // 2  # 2时候，后面补1位， 4时候补两位...
        z = torch.zeros_like(len_idx)
        z[: -tail_padding] = len_idx[tail_padding:]
        anchor_idx = torch.cat([previous_anchor_idx, z], dim=-1)  # [seqlen, anchor_len]
    #idx = anchor_idx.unsqueeze(0).expand(batch_size, anchor_idx.size()[0], anchor_idx.size()[1])
    anchor_repr = word_repr[batch_idx, anchor_idx]
    return anchor_idx, anchor_repr

def enum_anchor_repr_fmt(word_repr, len_idx, anchor_num, previous_anchor_idx, batch_idx, batch_size):
    if anchor_num == 1:
        anchor_idx = len_idx
    elif anchor_num % 2 == 1:  # 补前位
        front_padding = (anchor_num - 1) // 2  # 3时候，前面补1位，5时候补两位...
        z = torch.zeros_like(len_idx)
        z[front_padding:] = len_idx[: -front_padding]
        anchor_idx = torch.cat([z, previous_anchor_idx], dim=-1)
    else:
        tail_padding = anchor_num // 2  # 2时候，后面补1位， 4时候补两位...
        z = torch.zeros_like(len_idx)
        z[: -tail_padding] = len_idx[tail_padding:]
        anchor_idx = torch.cat([previous_anchor_idx, z], dim=-1)  # [seqlen, anchor_len]
    idx = torch.cat([anchor_idx[:, 0].unsqueeze(1), len_idx, anchor_idx[:, -1].unsqueeze(1)], dim=-1). \
        unsqueeze(0).expand(batch_size, anchor_idx.size()[0], 3)
    anchor_repr = word_repr[batch_idx, idx]
    return anchor_idx, anchor_repr

def enum_anchor_repr_fmtA(word_repr, len_idx, anchor_num, previous_anchor_idx, batch_idx, batch_size):
    if anchor_num == 1:
        anchor_idx = len_idx
    elif anchor_num % 2 == 1:  # 补前位
        front_padding = (anchor_num - 1) // 2  # 3时候，前面补1位，5时候补两位...
        z = torch.zeros_like(len_idx)
        z[front_padding:] = len_idx[: -front_padding]
        anchor_idx = torch.cat([z, previous_anchor_idx], dim=-1)
    else:
        tail_padding = anchor_num // 2  # 2时候，后面补1位， 4时候补两位...
        z = torch.zeros_like(len_idx)
        z[: -tail_padding] = len_idx[tail_padding:]
        anchor_idx = torch.cat([previous_anchor_idx, z], dim=-1)  # [seqlen, anchor_len]
    anchor_repr_list = [anchor_idx[:, 0].unsqueeze(1)]
    if anchor_num == 2:
        anchor_repr_list += [anchor_idx[:, -1].unsqueeze(1)]
    elif anchor_num >= 3:  # first current and tail
        anchor_repr_list += [len_idx, anchor_idx[:, -1].unsqueeze(1)]

    idx = torch.cat(anchor_repr_list, dim=-1). \
        unsqueeze(0).expand(batch_size, anchor_idx.size()[0], len(anchor_repr_list))
    anchor_repr = word_repr[batch_idx, idx]
    return anchor_idx, anchor_repr

def get_anchor_repr(hyps, anchor_const, BATCH_SIZE, SEQ_LEN, repr_dim, representation):
    """construct representation for classification"""
    if hyps["anchor_repr"] == 'ALL':
        anchor_repr_depth = anchor_const
        enum_anchor_func = enum_anchor_repr
    elif hyps["anchor_repr"] == 'FMT':
        print("Anchor repr use FMT") if consts.ONECHANCE else None
        anchor_repr_depth = 3
        enum_anchor_func = enum_anchor_repr_fmt
    else:
        print("Anchor repr use FMTA") if consts.ONECHANCE else None
        anchor_repr_depth = 3
        enum_anchor_func = enum_anchor_repr_fmtA

    anchor_representation = torch.zeros(BATCH_SIZE, SEQ_LEN, anchor_const, anchor_repr_depth,
                                               repr_dim).cuda()
    batch_matrix = np.array([range(BATCH_SIZE)] * SEQ_LEN).transpose()
    batch_id = torch.LongTensor(batch_matrix).contiguous().view(BATCH_SIZE, SEQ_LEN, 1).cuda()
    previous_anchor_idx = None
    len_idx = torch.LongTensor(range(SEQ_LEN)).unsqueeze(1).cuda()
    for anchor_num in range(1, anchor_const + 1):
        anchor_idx, anchor_repr = enum_anchor_func(word_repr=representation, len_idx=len_idx, anchor_num=
                                                   anchor_num, previous_anchor_idx=previous_anchor_idx,
                                                   batch_idx=batch_id, batch_size=BATCH_SIZE)
        if hyps["anchor_repr"] == 'ALL':
            anchor_representation[:, :, anchor_num - 1, :anchor_num, :] = anchor_repr
        elif hyps["anchor_repr"] == 'FMT':
            anchor_representation[:, :, anchor_num - 1, :anchor_repr_depth, :] = anchor_repr
        else:
            anchor_representation[:, :, anchor_num - 1, :min(anchor_repr_depth, anchor_num), :] = anchor_repr
        previous_anchor_idx = anchor_idx
    return anchor_representation