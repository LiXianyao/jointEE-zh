import torch
import torch.nn as nn

from util.util import log


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.hyperparams = None
        self.device = torch.device("cpu")

    def __getnewargs__(self):
        # for pickle
        return self.hyperparams

    def __new__(cls, *args, **kwargs):
        log('created %s with params %s' % (str(cls), str(args)))

        instance = super(Model, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

    def test_mode_on(self):
        self.test_mode = True
        self.eval()

    def test_mode_off(self):
        self.test_mode = False
        self.train()

    def bert_parameters_requires_grads(self, args):  # "with bert"
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "bertFF" in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.l2decay},
            {'params': [p for n, p in param_optimizer if "bertFF" in n and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        print("model with bert parameters {}".format(len(optimizer_grouped_parameters[0]['params']) +
                                                       len(optimizer_grouped_parameters[1]['params'])))
        return optimizer_grouped_parameters

    def parameters_requires_grads(self, args):  # "without bert"
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'bn', 'bert']
        prefix_exclude = ['bert']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "bertFF" not in n and not any(nd in n for nd in no_decay)],
             'weight_decay': args.l2decay},
            {'params': [p for n, p in param_optimizer if "bertFF" not in n and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        print("model with normal parameters {}".format(len(optimizer_grouped_parameters[0]['params']) +
                                                       len(optimizer_grouped_parameters[1]['params'])))
        return optimizer_grouped_parameters

    def parameters_requires_grad_clipping(self):
        return list(filter(lambda pram: pram.requires_grad, self.parameters()))

    def save_model(self, path):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
