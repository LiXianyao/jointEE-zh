
import torch
import torch.nn as nn
from util.util import *
from util import consts
from transformers import BertModel
import copy

class BertRepresentationLayer(nn.Module):

    def __init__(self, hyps, bert_model, bert_config):
        super(BertRepresentationLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.bert = BertModel.from_pretrained(bert_model, config=bert_config)
        self.dropout = nn.Dropout(hyps["bert_dropout"])

    def forward(self, tokens, token_mask, segments, token_head_mask):

        embedding_output = self.bert.embeddings(tokens, segments[0])
        embedding_output = self.dropout(embedding_output)
        # feed into bert
        if self.training:
            self.bert.train()
            sequence_output, pooled_output, bert_encode_layers = self.bert(token_type_ids=segments[0], attention_mask=token_mask, inputs_embeds=embedding_output)
        else:
            with torch.no_grad():
                sequence_output, pooled_output, bert_encode_layers = self.bert(token_type_ids=segments[0], attention_mask=token_mask, inputs_embeds=embedding_output)

        print("size of sequence_output is {}x{}".format(len(sequence_output), sequence_output.size())) if consts.ONECHANCE else None
        print("size of bert_encode_layers is {}x{}".format(len(bert_encode_layers), bert_encode_layers[
            0].size())) if consts.ONECHANCE else None

        trigger_hidden = self.pick_one_layer(bert_encode_layers, layer_num=-self.hyperparams["bert_layers"],
                                             token_head_mask=token_head_mask)
        entity_hidden = self.pick_one_layer(bert_encode_layers, layer_num=-self.hyperparams["bert_layers"] + 1,
                                             token_head_mask=token_head_mask)

        return trigger_hidden, entity_hidden  #, logits

    def pick_one_layer(self, bert_encode_layers, layer_num, token_head_mask):
        hidden = bert_encode_layers[layer_num]  # (x_len, 4*768)
        print("size of bert_encode_layers is {}x{}".format(len(bert_encode_layers), bert_encode_layers[0].size())) if consts.ONECHANCE else None
        return self.merge_emb(hidden, token_head_mask)

    def merge_emb(self, token_repr_emb, token_head_mask):
        token_repr_emb = torch.transpose(token_repr_emb, 1, 2)  # [batch, dim, token_len]
        print("token_repr size is {}, mask_size is {}".format(token_repr_emb.size(), token_head_mask.size())) if consts.ONECHANCE else None
        word_repr_output = torch.matmul(token_repr_emb, token_head_mask)  # [batch, dim, word_len]
        word_repr_output = torch.transpose(word_repr_output, 1, 2)  # [batch, word_len, dim]
        return word_repr_output

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))