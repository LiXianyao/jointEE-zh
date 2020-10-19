import copy

from .EmbeddingLayer import *
from util.util import *
from util import consts, tensorSetOp
from .CandidateRepresentationLayer import CandidateRepresentationLayer
from util import consts

class ArgumentRoleClsLayer(nn.Module):
    def __init__(self, hyps, input_size, class_num, weight):
        super(ArgumentRoleClsLayer, self).__init__()
        self.hyperparams = copy.deepcopy(hyps)
        self.cuda()
        self.arg_cls_num = class_num
        self.trigger_cls_num = hyps["trigger_label_weight"].size()[0]
        self.entity_cls_num = 2  # hyps["entity_label_weight"].size()[0]
        if not self.hyperparams["ARG_enable"]:
            return

        self.trigger_candidateRepresentationLayer = CandidateRepresentationLayer(hyps, strategy=hyps["arg_strategy"],
                                                                                 anchor_num=consts.TRIGGER_ANCHOR_NUM)
        self.entity_candidateRepresentationLayer = CandidateRepresentationLayer(hyps, strategy=hyps["arg_strategy"],
                                                                                anchor_num=consts.ENTITY_ANCHOR_NUM)
        self.trigger_sample_number = hyps["trigger_candidate_num"]
        self.entity_sample_number = hyps["entity_candidate_num"]
        # trigger_type Embedding Layer
        self.trigger_cls_embeddings = OneHotEmbeddingLayer(embedding_size=self.trigger_cls_num,
                                                           dropout=hyps["dropout"])
        #self.trigger_cls_embeddings = EmbeddingLayer(embedding_size=[self.trigger_cls_num, input_size * 2],
        #                                             dropout=hyps["dropout"])
        # entity_type Embedding Layer
        self.entity_cls_embeddings = OneHotEmbeddingLayer(embedding_size=self.entity_cls_num,
                                                           dropout=hyps["dropout"])
        input_size = input_size * 2 + self.trigger_cls_num  # + self.entity_cls_num
        self.linear = torch.nn.Linear(in_features=input_size, out_features=self.arg_cls_num, bias=True)
        init_linear_(self.linear)

        #self.sizeof("rpn linear", self.linear.weight)
        self.loss = nn.MultiMarginLoss(p=1, margin=5, weight=weight)
        self.dropout = nn.Dropout(hyps["dropout"])

    def forward(self, seq_mask,
                trigger_anchor_loc, trigger_anchor_type, trigger_predict_labels, trigger_representation, trigger_can_idx,
                entity_anchor_loc, entity_anchor_type, entity_predict_labels, entity_representation, entity_can_idx,
                trigger_label_i2s, trigger_label_s2i, entity_label_i2s, batch_golden_events):

        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_events = [{} for _ in range(BATCH_SIZE)]
        if not self.hyperparams["ARG_enable"]:
            return zero_loss, zero_events

        pos_cls_idx = consts.PADDING_IDX if trigger_label_i2s[consts.PADDING_IDX] == consts.PADDING_LABEL else 0
        trigger_candidates_idx = trigger_can_idx  # 0 and 1 should be other and padding cls
        golden_trigger_idx = torch.nonzero(trigger_anchor_type > pos_cls_idx)

        entity_candidates_idx = entity_can_idx
        golden_entity_idx = torch.nonzero(entity_anchor_type == 1)

        trigger_labels = trigger_predict_labels
        entity_labels = entity_predict_labels
        if self.training:
            # when training, use the golden trigger and entities that not include in the candidates
            trigger_candidates_idx = tensorSetOp.indexSetOp3d(trigger_candidates_idx, golden_trigger_idx,
                                                              operator="union")
            entity_candidates_idx = golden_entity_idx
            trigger_labels = trigger_anchor_type
            entity_labels = entity_anchor_type
        elif self.hyperparams["golden_arg"]:
            print("test with golden arg {}".format(bool(self.hyperparams["golden_arg"]))) if consts.ONECHANCE else None
            """only use golden to test —— to confirm the upper bound of this module"""
            trigger_candidates_idx = golden_trigger_idx
            entity_candidates_idx = golden_entity_idx
            trigger_labels = trigger_anchor_type
            entity_labels = entity_anchor_type

        trigger_candidates_repr, trigger_candidates_label, trigger_candidate_len, \
            trigger_candidate_loc = self.trigger_candidateRepresentationLayer.forward(
                trigger_representation,
                trigger_candidates_idx,
                trigger_anchor_loc,
                trigger_labels)

        entity_candidates_repr, entity_candidates_label, entity_candidate_len, \
            entity_candidate_loc = self.entity_candidateRepresentationLayer.forward(
                entity_representation,
                entity_candidates_idx,
                entity_anchor_loc,
                entity_labels)

        ae_input = []
        ae_logits_key = []
        for sid in range(BATCH_SIZE):
            sen_trigger_num = trigger_candidate_len[sid]
            sen_entity_num = entity_candidate_len[sid]
            if not sen_trigger_num or not sen_entity_num: continue
            batch_trigger_label = torch.stack(trigger_candidates_label[sid], dim=0)  # [can_num]
            batch_trigger_label_emb = self.trigger_cls_embeddings(batch_trigger_label)  # [can_num, dim]

            #batch_entity_label = torch.stack(entity_candidates_label[sid], dim=0)
            #batch_entity_label_emb = self.entity_cls_embeddings(batch_entity_label)  # [can_num, dim]
            #print("size of batch entitys' label is {}".format(batch_entity_label.size()))
            #print("size of batch entitys' label emb is {}".format(batch_entity_label_emb.size()))
            for tid in range(sen_trigger_num):
                trigger_repr = trigger_candidates_repr[sid][tid]
                t_sid, t_eid = trigger_candidate_loc[sid][tid].tolist()[:2]
                trigger_type = trigger_label_i2s[trigger_candidates_label[sid][tid].item()]
                trigger_emb = batch_trigger_label_emb[tid]

                for eid in range(sen_entity_num):
                    entity_repr = entity_candidates_repr[sid][eid]
                    e_sid, e_eid = entity_candidate_loc[sid][eid].tolist()[:2]
                    entity_type = entity_label_i2s[entity_candidates_label[sid][eid].item()]
                    #entity_emb = batch_entity_label_emb[eid]

                    pair_repr = torch.cat([trigger_repr, entity_repr])  # 2 * REP_DIM
                    self.add_ae_repr(pair_repr, trigger_type_emb=trigger_emb,
                                     entity_type_emb=None, ae_input=ae_input)
                    ae_logits_key.append((sid, t_sid, t_eid, trigger_type, e_sid, e_eid, entity_type))
        if len(ae_input) != 0:  # the linear layer forbid the input of size 0 at dim 0
            ae_hidden = self.dropout(torch.stack(ae_input, dim=0))
            ae_logits = self.linear(ae_hidden)
            print("ae_logits 's size is {}".format(
                ae_logits.size())) if consts.ONECHANCE else None

            loss_ae, predicted_events = self.calculate_loss_ae(ae_logits, ae_logits_key, batch_golden_events, BATCH_SIZE, trigger_label_s2i)
        else:
            loss_ae = zero_loss
            predicted_events = zero_events
        return loss_ae, predicted_events

    def add_ae_repr(self, pair_repr, trigger_type_emb, entity_type_emb, ae_input):
        if self.hyperparams["arg_gate"]:
            pass
            #pair_repr_ = self.arg_gate_repr(pair_repr, trigger_type_emb, entity_type_emb)
        else:
            if trigger_type_emb is not None:
                pair_repr = torch.cat([pair_repr, trigger_type_emb], dim=-1)
            if entity_type_emb is not None:
                pair_repr = torch.cat([pair_repr, entity_type_emb], dim=-1)
        ae_input.append(pair_repr)

    def calculate_loss_ae(self, logits, keys, batch_golden_events, BATCH_SIZE, trigger_label_s2i):
        '''
        Calculate loss for a batched output of ae

        :param logits: FloatTensor, (N, output_class)
        :param keys: [(i, st, ed, trigger_type_str, e_st, e_ed, e_type_str), ...]
        :param batch_golden_events:
        [
            {
                (2, 3, "event_type_str") --> [(1, 2, XX), ...]
                , ...
            }, ...
        ]
        :param BATCH_SIZE: int
        :return:
            loss: Float, accumulated loss and index
            predicted_events:
            [
                {
                    (2, 3, "event_type_str") --> [(1, 2, XX), ...]
                    , ...
                }, ...
            ]
        '''
        # print(batch_golden_events)
        golden_labels = []
        sample_idx = []  # 采样所有含有触发词的句子的样本
        for kid in range(len(keys)):   # entity_type is not used here
            i, st, ed, event_type_str, e_st, e_ed, entity_type = keys[kid]
            label = consts.ROLE_O_LABEL
            #if len(batch_golden_events[i]) > 0:
            #    sample_idx.append(kid)
            if (st, ed, event_type_str) in batch_golden_events[i]:  # if event matched  event_type正确的情况下才看entity
                #print("match event, batch={}, s {}, e{}, type{}".format(i, st, ed, event_type_str))
                #print("with entity", batch_golden_events[i][(st, ed, event_type_str)])
                #print("target:", e_st, e_ed, entity_type)
                #sample_idx.append(kid)  # only sample the data with correct trigger
                for e_st_, e_ed_, r_label, e_type in batch_golden_events[i][(st, ed, event_type_str)]:  # 遍历这个事件的golden entities
                    #self.update_arg_mask(trigger_idx=trigger_label_s2i[event_type_str], arg_idx=r_label)
                    if e_st == e_st_ and e_ed == e_ed_:  # 匹配上了则设置其label
                        #print("match entity", e_st, e_ed, r_label)
                        label = r_label
                        break
            golden_labels.append(label)   # 对每个match上的事件，取出其对应每个entity的label结果，若事件不对，则直接定label为OTHER
        golden_labels = torch.LongTensor(golden_labels).cuda()

        loss = self.loss(logits, golden_labels)

        predicted_events = [{} for _ in range(BATCH_SIZE)]  # 构造词典，保存每个（正确的）trigger候选的分类正确的argument
        output_ae = torch.max(logits, 1)[1].view(golden_labels.size()).tolist()
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), ae_label in zip(keys, output_ae):
            if ae_label == consts.ROLE_O_LABEL: continue
            if (st, ed, event_type_str) not in predicted_events[i]:
                predicted_events[i][(st, ed, event_type_str)] = []
            predicted_events[i][(st, ed, event_type_str)].append((e_st, e_ed, ae_label, entity_type))

        return loss, predicted_events

    def sizeof(self, name, tensor):
        if not consts.VISIABLE: return
        log("shape of tensor '{}' is {} ".format(name, tensor.size()))
