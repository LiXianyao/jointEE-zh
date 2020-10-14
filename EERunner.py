#-*-encoding:utf-8-*-#
import argparse
import os
import json
import sys
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import Field

from util import consts
from util.corpus.Data import ACE2005Dataset
from util.corpus.Field import MultiTokenField, MultiTokenTensorField, EventField, EntityField, TokenField
from util.util import log
from util.Configure import *

from EETester import EETester
from Trainer import Trainer

class EERunner(object):
    def __init__(self, conf, device=None):
        conf.__dict__.update(conf.confs)
        self.arg = conf
        self.arg.hps = conf.confs
        if device:
            self.arg.device = device

        self.bertConfig = None
        self.bertTokenizer = None
        self.modelClass = None
        self.train = os.path.join(self.arg.home_path, self.arg.data_path, "train.json")
        self.test = os.path.join(self.arg.home_path, self.arg.data_path, "test.json")
        self.dev = os.path.join(self.arg.home_path, self.arg.data_path, "dev.json")

    def set_device(self, device="cpu"):
        self.device = torch.device(device)
        if device != "cpu":
            torch.cuda.set_device(self.device.index)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def load_model(self, existModel):
        """ :param existModel model name that previously trained"""
        config, un_used = self.bertConfig.from_pretrained(self.arg.bert_model, num_labels=self.arg.hps["trigger_BIOTag_num"], cache_dir="../../pretrain-bert/{}/".format(self.arg.bert_model),
                                              output_hidden_states=True, return_unused_kwargs=True)
        print("unused args = {}".format(un_used))
        myModel = self.modelClass(self.arg.hps, device=self.device, bert_model=self.arg.bert_model,
                                from_tf=False, config=config)
        if existModel is not None:
            myModel.load_model(existModel)
        if self.get_device().type == "cuda":
            myModel = myModel.cuda(self.get_device().index)
        return myModel

    def get_tester(self):
        return EETester()

    def run(self, Train=True, Test=False):
        print("Running on", self.arg.device)
        self.set_device(self.arg.device)

        np.random.seed(self.arg.seed)
        torch.manual_seed(self.arg.seed)

        # create training set
        if self.arg.data_path:
            log('loading corpus from %s' % self.arg.data_path)
        if not os.path.exists(self.arg.output_path):
            os.makedirs(self.arg.output_path)
            os.makedirs(self.arg.output_path + "/code")
            os.makedirs(self.arg.output_path + "/code/models")

        self.define_input_field()  # define the fields of several inputs
        print(self.arg)
        print(self.bertConfig)
        print(self.bertTokenizer)

        consts.TOKEN_MASK_TYPE = self.arg.token_mask_type
        self.train_set = self.construct_dataset(self.train, keep_events=1, skip_sample=self.arg.skip_sample, tokenizer=self.bertTokenizer)  # load datafiles and transinto field
        self.dev_set = self.construct_dataset(self.dev, tokenizer=self.bertTokenizer)
        self.test_set = self.construct_dataset(self.test, tokenizer=self.bertTokenizer)
        self.buil_field_vocab()  # build vocab on train and dev set
        tester = self.get_tester()


        if self.arg.restart > 0:
            log('init model from ' + self.arg.demo_model)
            self.model = self.load_model(self.arg.demo_model)
            log('model loaded, there are %i sets of params' % len(self.model.parameters_requires_grad_clipping()))
        else:
            self.model = self.load_model(None)
            log('model created from scratch, there are %i sets of params' % len(self.model.parameters_requires_grad_clipping()))

        self.arg.word_i2s = self.WordsField.vocab.itos
        self.arg.trigger_label_i2s = self.TriggerLabelField.vocab.itos
        optimizer_constructor, bert_optimizer_constructor = self.get_otimizer_constructor(self.model)
        trainer = Trainer(model=self.model, args=self.arg, word_i2s=self.arg.word_i2s, EERuner=self,
                          optimizer_constructor=optimizer_constructor,
                          bert_optimizer_constructor=bert_optimizer_constructor, tester=tester)
        if Train:
            print("backup codes")
            os.system("cp config.cfg {}".format(self.arg.output_path))
            os.system("cp network/models/*.py {}".format(self.arg.output_path + "/code/models"))
            self.store_vec()
            train_writer = SummaryWriter(os.path.join(self.arg.output_path, "train"))
            detection_writer = SummaryWriter(os.path.join(self.arg.output_path, "detection"))
            classification_writer = SummaryWriter(os.path.join(self.arg.output_path, "classification"))
            self.arg.writer = {"train": train_writer, "detect": detection_writer, "cls": classification_writer}
            trainer.train(train_set=self.train_set, dev_set=self.dev_set, test_set=self.test_set, epochs=
                          self.arg.epochs, other_testsets={})
            self.arg.writer["train"].close()
            self.arg.writer["detect"].close()
            self.arg.writer["cls"].close()
        if Test:
            trainer.eval(test_set=self.test_set)

        log('Done!')

    def get_otimizer_constructor(self, model):
        model_parameters = model.parameters_requires_grads(self.arg)
        bert_parameters = model.bert_parameters_requires_grads(self.arg)

        if self.arg.optimizer == "adadelta":
            optimizer_constructor = partial(torch.optim.Adadelta, params=model_parameters, weight_decay=self.arg.l2decay)

        elif self.arg.optimizer == "adam":
            optimizer_constructor = partial(torch.optim.Adam, params=model_parameters,  weight_decay=self.arg.l2decay)

        else:
            optimizer_constructor = partial(torch.optim.SGD, params=model_parameters, weight_decay=self.arg.l2decay,
                                            momentum=0.9)

        bert_optimizer_constructor = partial(torch.optim.Adam, params=bert_parameters, weight_decay=0.)
        return optimizer_constructor, bert_optimizer_constructor

    def recover_data(self, batch_data, length, field, depth):
        """
        recover tensor of the batchfy-field's data into string
        only use for checking
         """
        data_size = batch_data.size()
        itos = field.vocab.itos
        recover_res = []
        for batch in range(data_size[0]):
            recover_res.append("")
            for seq_len in range(data_size[1]):
                word = ""
                if depth == 3:
                    for idx in range(data_size[2]):
                        if idx >= length[batch][seq_len]: continue
                        word += itos[batch_data[batch, seq_len, idx]]
                else:
                    if seq_len >= length[batch]: continue
                    word = itos[batch_data[batch, seq_len]]
                recover_res[-1] += " " + word
        return recover_res

    def define_input_field(self):
        self.WordsField = Field(lower=True, include_lengths=True, batch_first=True)
        self.PosTagsField = Field(lower=True, batch_first=True, fix_length=consts.CUTOFF)
        self.WordsTokensField = TokenField(lower=True, include_lengths=True, use_vocab=True, batch_first=True,
                                           pad_token=consts.PADDING_LABEL, unk_token=consts.UNK_LABEL)
        self.WordsTokensValidField = MultiTokenTensorField(include_lengths=True, use_vocab=False, batch_first=True, dtype=torch.float32,
                                                           pad_token=0., fix_length=(consts.CUTOFF, None))
        self.EntityAnchorLabelsField = MultiTokenTensorField(lower=False, use_vocab=False, batch_first=True,
                                                             pad_token=0, unk_token=0, fix_length=(consts.CUTOFF, None))
        self.EntityAnchorTypesField = MultiTokenTensorField(lower=False, batch_first=True, use_vocab=True,
                                                            pad_token="OTHER", unk_token="OTHER", fix_length=(consts.CUTOFF, None))  # 长度为k的label
        self.EntityAnchorField = MultiTokenTensorField(lower=False, batch_first=True, use_vocab=False
                                                 ,pad_token=[-1, -1], unk_token=None, fix_length=(consts.CUTOFF, None))

        #AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
        self.TriggerLabelField = Field(lower=False, include_lengths=True, batch_first=True, pad_token="O", unk_token="O", use_vocab=True, fix_length=consts.CUTOFF)  # BIO标注的label
        self.TriggerAnchorLabelsField = MultiTokenTensorField(lower=False, use_vocab=False, batch_first=True,
                                                              pad_token=0, unk_token=0, fix_length=(consts.CUTOFF, None))
        self.TriggerAnchorTypesField = MultiTokenTensorField(lower=False, batch_first=True, use_vocab=True,
                                                             pad_token="OTHER", unk_token="OTHER", fix_length=(consts.CUTOFF, None))  # 长度为k的label
        self.TriggerAnchorField = MultiTokenTensorField(lower=False, batch_first=True, use_vocab=False
                                                 ,pad_token=[-1, -1], unk_token=None, fix_length=(consts.CUTOFF, None))
        self.EventsField = EventField(lower=False, batch_first=True, unk_token="OTHER")
        #self.EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)

    def store_vec(self):
        if not os.path.exists(self.arg.output_path):
            os.mkdir(self.arg.output_path)
        with open(os.path.join(self.arg.output_path, "entityAnchorType.json"), "w", encoding='utf-8') as f:
            json.dump(self.EntityAnchorTypesField.vocab.stoi, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.arg.output_path, "triggerlabel.json"), "w", encoding='utf-8') as f:
            json.dump(self.TriggerLabelField.vocab.stoi, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.arg.output_path, "tirggerAnchorType.json"), "w", encoding='utf-8') as f:
            json.dump(self.TriggerAnchorTypesField.vocab.stoi, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.arg.output_path, "role.json"), "w", encoding='utf-8') as f:
            json.dump(self.EventsField.vocab.stoi, f, ensure_ascii=False, indent=4)

    def construct_dataset(self, path, tokenizer, keep_events=0, only_keep=False, skip_sample=1):
        data_set = ACE2005Dataset(path=path, tokenizer=tokenizer,
                                   fields={"words": ("WORDS", self.WordsField),
                                           "pos-tags": ("POSTAGS", self.PosTagsField),
                                           "tokens": ("TOKENS", self.WordsTokensField),
                                           "tokens_mask": ("TOKENSMASK", self.WordsTokensValidField),
                                           #"golden-entity-mentions": ("ENTITYLABELS", self.EntityLabelsField),
                                           "entity-anchor": ("ENTITYANCHOR", self.EntityAnchorField),
                                           "entity-anchor-label": ("ENTITYANCHORLABEL", self.EntityAnchorLabelsField),
                                           "entity-anchor-class": ("ENTITYANCHORCLS", self.EntityAnchorTypesField),

                                           "golden-event-mentions": ("TRIGGERLABEL", self.TriggerLabelField),
                                           "trigger-anchor": ("TRIGGERANCHOR", self.TriggerAnchorField),
                                           "trigger-anchor-label": ("TRIGGERANCHORLABEL", self.TriggerAnchorLabelsField),
                                           "trigger-anchor-class": ("TRIGGERANCHORCLS", self.TriggerAnchorTypesField),
                                           "all-events": ("EVENTS", self.EventsField),
                                           #"all-entities": ("ENTITIES", self.EntitiesField)
                                           },
                                   keep_events=keep_events, only_keep=only_keep, skip_sample=skip_sample)
        print("{} set length {}".format(path, len(data_set)))
        return data_set

    def set_hyps(self):
        if "wemb_size" not in self.arg.hps:
            self.arg.hps["wemb_size"] = len(self.WordsField.vocab.itos)
        if "pemb_size" not in self.arg.hps:
            self.arg.hps["pemb_size"] = len(self.PosTagsField.vocab.itos)
        if "psemb_size" not in self.arg.hps:
            self.arg.hps["psemb_size"] = max([self.train_set.longest(), self.dev_set.longest(), self.test_set.longest()]) + 2
        if "entity_label_num" not in self.arg.hps:
            self.arg.hps["entity_label_num"] = len(self.EntityAnchorTypesField.vocab.itos)
        if "trigger_BIOTag_num" not in self.arg.hps:
            #print("Triggers Types are: {}".format(self.TriggerLabelField.vocab.itos))
            self.arg.hps["trigger_BIOTag_num"] = len(self.TriggerLabelField.vocab.itos)
        if "trigger_type_num" not in self.arg.hps:
            self.arg.hps["trigger_type_num"] = len(self.TriggerAnchorTypesField.vocab.itos)
        if "argument_type_num" not in self.arg.hps:
            self.arg.hps["argument_type_num"] = len(self.EventsField.vocab.itos)
        with open(os.path.join(self.arg.output_path, "param"), "w") as w:
            json.dump(self.arg.hps, w)
        #{'ae_oc': 36, 'eemb_size': 29, 'cemb_size': 90, 'wemb_size': 16599, 'oc': 57, 'pemb_size': 47, 'psemb_size': 52}

    def buil_field_vocab(self):
        """compute vocab for fields"""
        self.WordsField.build_vocab(self.train_set.WORDS, self.dev_set.WORDS, self.test_set.WORDS)
        self.PosTagsField.build_vocab(self.train_set.POSTAGS)
        self.WordsTokensField.build_vocab(self.bertTokenizer, self.train_set.TOKENS)
        self.EntityAnchorTypesField.build_vocab(self.train_set.ENTITYANCHORCLS)  # , self.dev_set.ENTITYANCHORCLS)
        self.TriggerLabelField.build_vocab(self.train_set.TRIGGERLABEL)  # , self.dev_set.TRIGGERLABEL)
        self.TriggerAnchorTypesField.build_vocab(self.train_set.TRIGGERANCHORCLS)  # , self.dev_set.TRIGGERANCHORCLS)
        self.EventsField.build_vocab(self.train_set.EVENTS)
        self.set_hyps()  # update the hyps with actual num of words
        consts.TRIGGER_ARG_MATRIX = torch.zeros([len(self.TriggerAnchorTypesField.vocab.itos),
                                                 len(self.EventsField.vocab.itos)]).float().cuda()
        consts.TRIGGER_ARG_MATRIX[:, 0] = 1.
        trigger_s2i = self.TriggerAnchorTypesField.vocab.stoi
        arg_s2i = self.EventsField.vocab.stoi
        for t_str in consts.TRIGGER_ARG_MAP:
            t_idx = trigger_s2i[t_str]
            for a_str in consts.TRIGGER_ARG_MAP[t_str]:
                a_idx = arg_s2i[a_str]
                consts.TRIGGER_ARG_MATRIX[t_idx, a_idx] = 1.
        #print(consts.TRIGGER_ARG_MAP)
        #print(consts.TRIGGER_ARG_MATRIX.tolist())

        #"""
        #set the alpha weight of TriggerType (except Other Type) into some float greater than 1.
        consts.ROLE_O_LABEL = self.EventsField.vocab.stoi["OTHER"]
        self.arg.hps["arg_label_weight"] = torch.ones([len(self.EventsField.vocab.itos)]) * self.arg.hps["arg_cls_weight"]
        self.arg.hps["arg_label_weight"][consts.ROLE_O_LABEL] = 1.0
        #print("O label for AE is", consts.ROLE_O_LABEL)

        consts.TRIGGER_O_LABEL = self.TriggerAnchorTypesField.vocab.stoi["OTHER"]
        self.arg.hps["trigger_label_weight"] = torch.ones([len(self.TriggerAnchorTypesField.vocab.itos)]) * self.arg.hps["trigger_cls_weight"]
        self.arg.hps["trigger_label_weight"][consts.TRIGGER_O_LABEL] = 1.0

        self.arg.hps["entity_label_weight"] = torch.ones([len(self.EntityAnchorTypesField.vocab.itos)])
        self.arg.hps["entity_label_weight"][0] = 1.0
        #"""
        return

if __name__ == "__main__":
    confs = Configure()
    runner = EERunner(confs)
    runner.run()
