
import numpy as np
import torch

def indexSetOp3d(index_tensor1, index_tensor2, operator):
    set1, set2 = set(), set()
    rowlen = 3  # means [batch, seqlen, anchor_num]
    for row in index_tensor1.tolist():
        set1.add(tuple(row))

    for row in index_tensor2.tolist():
        set2.add(tuple(row))

    if operator == "diff":
        res = list(set1 - set2)
    elif operator == "inter":
        res = list(set1 & set2)
    else:  # union
        set_diff = list(set2 - set1)
        set1_list = list(set1)
        minlen = min(len(set_diff), len(set1_list))
        res = []
        for (tup3, tup3_) in zip(set1_list[:minlen], set_diff[:minlen]):
            res.append(tup3)
            res.append(tup3_)
        res += set1_list[minlen:] + set_diff[minlen:]

    res_tensor = torch.zeros([len(res), rowlen], dtype=torch.int64, requires_grad=False)
    for idx in range(len(res)):
        for j in range(rowlen):
            res_tensor[idx][j] = res[idx][j]
    return res_tensor.cuda()
