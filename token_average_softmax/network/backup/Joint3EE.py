import copy

import numpy
import torch
from torch import nn
from .EventDetectionLayer import  EventDetectionLayer
from .EntityMentionDetectionLayer import EntityMentionDetectionLayer
from util.util import *
from util import consts
from .model import Model
from .BertRepresentationLayer import BertRepresentationLayer
from .ArgumentRoleClsLayer import ArgumentRoleClsLayer
from .inception import InceptionCNN

class Joint3EEModel(Model):
    def __init__(self, hyps, device, bert_model, *inputs, **kwargs):
        super(Joint3EEModel, self).__init__(*inputs, **kwargs)
        self.cuda(device)
        self.hyperparams = copy.deepcopy(hyps)
        self.device = device
        self.bertConfig = kwargs["config"]
        self.bertRepresentationLayer = BertRepresentationLayer(hyps, bert_model, self.bertConfig)

        bert_output_size = self.bertConfig.hidden_size
        repr_len = bert_output_size

        self.inception_repr_trigger = InceptionCNN(hyps, out_channels=hyps["inception_channel"],
                                                 kernel_dim=bert_output_size,
                                                 inception_mode=hyps["inception_mode"])

        self.inception_repr = InceptionCNN(hyps, out_channels=hyps["inception_channel"],
                                                 kernel_dim=bert_output_size,
                                                 inception_mode=hyps["inception_mode"])

        self.inception_repr_2 = InceptionCNN(hyps, out_channels=hyps["inception_channel"],
                                                 kernel_dim=bert_output_size,
                                                 inception_mode=hyps["inception_mode"])

        self.eventDetectionLayer = EventDetectionLayer(hyps, trigger_repr_len=repr_len)
        self.entityMentionDetectionLayer = EntityMentionDetectionLayer(hyps, entity_reprensentation_dim=repr_len)
        self.argumentRoleClsLayer = ArgumentRoleClsLayer(hyps, input_size=self.eventDetectionLayer.out_dim
                                                         , class_num=hyps["argument_type_num"], weight=hyps["arg_label_weight"])



    def forward(self, input_ids, input_len, postags, segments=None, trigger_labels=None, token_head_mask=None,
                trigger_labels_len=None, trigger_anchor_loc=None, trigger_anchor_labels=None, trigger_anchor_type=None,
                entity_anchor_loc=None, entity_anchor_labels=None, entity_anchor_type=None,
                trigger_label_i2s=None, trigger_label_s2i=None, entity_label_i2s=None, batch_golden_events=None):
        '''
        extracting event triggers

        :param word_sequence: LongTensor, padded word indices, (batch_size, seq_len)
                :param w_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size,)

        :param char_sequence: LongTensor, padded char indices, (batch_size, seq_len, max_word_len)
        :param c_len: numpy int64 array, indicating corresponding actual sequence length, (batch_size, max_word_len)

        :param pos_tagging_sequence: LongTensor, padded pos-tagging label indices, (batch_size, seq_len)
        :param entity_anchors_label: label (is entity or not) for every mention candidates, (batch_size, seq_len, ENTITY_ANCHOR_NUM)
        :param entity_anchors_cls: label (entity label) for every mention candidates, (batch_size, seq_len, ENTITY_ANCHOR_NUM)

        :param trigger_anchor_label: label (is trigger or not) for every trigger candidates, (batch_size, seq_len, TRIGGER_ANCHOR_NUM)
        :param trigger_anchor_cls: label (trigger label) for every trigger candidates, (batch_size, seq_len, TRIGGER_ANCHOR_NUM)
        :return:
            logits: FloatTensor, output logits of ED, (batch_size, seq_len, output_class)
            word_mask: ByteTensor, mask of input sequence, (batch_size, seq_len)
            ae_hidden: FloatTensor, output logits of AE, (N, output_class) or [] indicating no need to predicting arguments
            ae_logits_key: [], indicating how the big batch is constructed or []
        '''
        # get bert's embeddings
        attention_mask = self.get_mask_tensor(input_ids, input_len, 2)  # 对token序列的mask
        trigger_hidden, entity_hidden = self.bertRepresentationLayer.forward(
            tokens=input_ids, segments=segments, token_mask=attention_mask,
            token_head_mask=token_head_mask
        )
        batch_size, max_len, feat_dim = trigger_hidden.shape
        #print(w_len)
        trigger_labels_mask = self.get_mask_tensor(trigger_labels, trigger_labels_len, depth=2)  # 对trigger的mask

        ed_inception_out = self.inception_repr_trigger(trigger_hidden)
        emd_inception_out = self.inception_repr_2(self.inception_repr(entity_hidden))
        #emd_inception_out = self.inception_repr(entity_hidden)

        #(batch_size, seq_len, out) -》( batch, seqlen,  anchor_num, 2)
        loss_emd, emd_det_label, loss_emd_cls, emd_cls_label, entity_anchor_representation, \
            entity_candidates_repr, entity_candidates_label, entity_candidates_predict, entity_candidates_num, \
            entity_candidates_len, \
            entity_candidates_mask, entity_candidates_loc = self.entityMentionDetectionLayer.forward(
                seq_mask=trigger_labels_mask, cnn_representation=emd_inception_out, word_representation=entity_hidden,
                entity_anchor_labels=entity_anchor_labels, entity_anchor_loc=entity_anchor_loc,
                entity_anchor_type=entity_anchor_type)

        loss_ed, ed_detect_label, loss_ed_cls, ed_cls_label, reg_trigger_representation, trigger_candidates_repr, \
            trigger_candidates_label, trigger_candidates_predict, trigger_candidates_num, trigger_candidates_len, \
            trigger_candidates_mask, trigger_candidates_loc, trigger_candidates_weight, key_candidate_att_repr = \
            self.eventDetectionLayer.forward(
                seq_mask=trigger_labels_mask, cnn_representation=ed_inception_out, word_representation=trigger_hidden,
                trigger_anchor_labels=trigger_anchor_labels, trigger_anchor_loc=trigger_anchor_loc,
                trigger_anchor_type=trigger_anchor_type,
                entity_candidates_repr=entity_candidates_repr, entity_candidates_mask=entity_candidates_mask,
                entity_candidates_len=entity_candidates_len, entity_candidates_loc=entity_candidates_loc)

        #"""
        loss_ae, predicted_events = self.argumentRoleClsLayer.forward(
            seq_mask=trigger_labels_mask,

            trigger_candidates_repr=trigger_candidates_repr, trigger_candidates_label=trigger_candidates_label,
            trigger_candidates_predict=trigger_candidates_predict, trigger_candidates_len=trigger_candidates_len,
            trigger_candidates_num=trigger_candidates_num,
            trigger_candidates_mask=trigger_candidates_mask, trigger_candidates_loc=trigger_candidates_loc,

            entity_candidates_repr=entity_candidates_repr, entity_candidates_label=entity_candidates_label,
            entity_candidates_predict=entity_candidates_predict,
            entity_candidates_len=entity_candidates_len, entity_candidates_num=entity_candidates_num,
            entity_candidates_mask=entity_candidates_mask, entity_candidates_loc=entity_candidates_loc,
            #entity_candidate_att_repr=entity_candidate_att_repr,

            trigger_label_i2s=trigger_label_i2s, entity_label_i2s=entity_label_i2s,
            batch_golden_events=batch_golden_events)
        """
        loss_ae, predicted_events = self.argumentRoleClsLayer.no_forward(
            seq_mask=trigger_labels_mask)
        #"""

        consts.ONECHANCE = False
        return loss_ed, ed_detect_label, loss_ed_cls, ed_cls_label, loss_emd, emd_det_label, loss_emd_cls, emd_cls_label\
            , loss_ae, predicted_events, trigger_candidates_weight

    def get_mask_tensor(self, sequence, length, depth):
        size = sequence.size()
        mask = numpy.zeros(shape=size, dtype=numpy.uint8)
        for i in range(size[0]):
            if depth == 2:
                s_len = int(length[i])
                mask[i, 0:s_len] = numpy.ones(shape=(s_len), dtype=numpy.uint8)
            elif depth == 3:
                for j in range(size[1]):
                    c_len = int(length[i][j])
                    mask[i, j, :c_len] = numpy.ones(shape=(c_len), dtype=numpy.uint8)
        mask = torch.tensor(mask, requires_grad=False).byte().cuda()
        return mask

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))
