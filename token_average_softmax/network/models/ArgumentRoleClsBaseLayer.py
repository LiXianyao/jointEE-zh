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
        input_size = input_size * 2 + self.trigger_cls_num + self.entity_cls_num
        self.linear = torch.nn.Linear(in_features=input_size, out_features=self.arg_cls_num, bias=True)
        init_linear_(self.linear)

        self.arg_mask_init = consts.TRIGGER_ARG_MATRIX
        self.arg_mask = consts.TRIGGER_ARG_MATRIX  # torch.nn.Parameter(self.arg_mask_init)
        #self.arg_mask[:, 0] = 1.
        self.update_mask = True

        #self.sizeof("rpn linear", self.linear.weight)
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.dropout = nn.Dropout(hyps["dropout"])

    def forward(self, seq_mask,
                trigger_anchor_loc, trigger_anchor_type, trigger_predict_labels, trigger_representation,
                entity_anchor_loc, entity_anchor_type, entity_predict_labels, entity_representation,
                trigger_label_i2s, trigger_label_s2i, entity_label_i2s, batch_golden_events):

        BATCH_SIZE, SEQ_LEN = seq_mask.size()[:2]
        zero_loss = torch.zeros([1]).cuda()
        zero_events = [{} for _ in range(BATCH_SIZE)]
        if not self.hyperparams["ARG_enable"]:
            return zero_loss, zero_events

        def shuffle_idx(input_idx, sample_num):
            if input_idx.size()[0] > 0:
                sample_array = np.array(range(input_idx.size()[0]))
                np.random.shuffle(sample_array)
                if sample_num is not None and sample_num > 0:
                    input_idx = input_idx[torch.tensor(sample_array[:sample_num])]  # sample_num x 2
                else:
                    input_idx = input_idx[torch.tensor(sample_array)]  # sample_num x 2
            return input_idx

        ## first, get all trigger candidates and entity candidates representation from the predicted labels
        pos_cls_idx = consts.PADDING_IDX if trigger_label_i2s[consts.PADDING_IDX] == consts.PADDING_LABEL else 0
        trigger_candidates_idx = shuffle_idx(torch.nonzero(trigger_predict_labels > pos_cls_idx),
                                             sample_num=BATCH_SIZE * self.trigger_sample_number)  # 0 and 1 should be other and padding cls
        golden_trigger_idx = torch.nonzero(trigger_anchor_type > pos_cls_idx)

        #pos_cls_idx = consts.PADDING_IDX if entity_label_i2s[consts.PADDING_IDX] == consts.PADDING_LABEL else 0
        #entity_candidates_idx = torch.nonzero(entity_predict_labels > consts.PADDING_IDX)
        entity_candidates_idx = shuffle_idx(torch.nonzero(entity_predict_labels == 1),
                                            sample_num=BATCH_SIZE * self.entity_sample_number)  # since EMD only detection, just pick the type 1 is ok
        golden_entity_idx = torch.nonzero(entity_anchor_type == 1)

        trigger_labels = trigger_predict_labels
        entity_labels = entity_predict_labels
        if self.training:
            # when training, use the golden trigger and entities that not include in the candidates
            trigger_candidates_idx = tensorSetOp.indexSetOp3d(trigger_candidates_idx, golden_trigger_idx,
                                                              operator="union")
            entity_candidates_idx = tensorSetOp.indexSetOp3d(entity_candidates_idx, golden_entity_idx,
                                                              operator="union")
            trigger_labels = trigger_anchor_type
            entity_labels = entity_anchor_type
        elif self.hyperparams["golden_arg"]:
            print("test with golden arg {}".format(bool(self.hyperparams["golden_arg"]))) if consts.ONECHANCE else None
            """only use golden to test —— to confirm the upper bound of this module"""
            trigger_candidates_idx = golden_trigger_idx
            entity_candidates_idx = golden_entity_idx
            trigger_labels = trigger_anchor_type
            entity_labels = entity_anchor_type

        #if not self.training: self.output_arg_mask()
        # [BATCH_SIZE, batch_candidate_num, REPR_DIM]
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
                entity_anchor_loc, entity_labels)

        ae_input = []
        ae_logits_key = []
        ae_logits_mask = None
        for sid in range(BATCH_SIZE):
            sen_trigger_num = trigger_candidate_len[sid]
            sen_entity_num = entity_candidate_len[sid]
            if not sen_trigger_num or not sen_entity_num: continue
            batch_trigger_label = torch.stack(trigger_candidates_label[sid], dim=0)  # [can_num]
            batch_trigger_label_emb = self.trigger_cls_embeddings(batch_trigger_label)  # [can_num, dim]
            batch_trigger_logits_mask = self.arg_mask[batch_trigger_label].view(sen_trigger_num, 1, self.arg_cls_num).\
                expand(sen_trigger_num, sen_entity_num, self.arg_cls_num).contiguous().\
                view(sen_trigger_num * sen_entity_num, self.arg_cls_num)

            batch_entity_label = torch.stack(entity_candidates_label[sid], dim=0)
            batch_entity_label_emb = self.entity_cls_embeddings(batch_entity_label)  # [can_num, dim]
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
                    entity_emb = batch_entity_label_emb[eid]

                    pair_repr = torch.cat([trigger_repr, entity_repr])  # 2 * REP_DIM
                    self.add_ae_repr(pair_repr, trigger_type_emb=trigger_emb,
                                     entity_type_emb=entity_emb, ae_input=ae_input)
                    ae_logits_key.append((sid, t_sid, t_eid, trigger_type, e_sid, e_eid, entity_type))
            if ae_logits_mask is None:
                ae_logits_mask = batch_trigger_logits_mask
            else:
                ae_logits_mask = torch.cat([ae_logits_mask, batch_trigger_logits_mask], dim=0)
        if len(ae_input) != 0:  # the linear layer forbid the input of size 0 at dim 0
            ae_logits = self.linear(torch.stack(ae_input, dim=0))
            if not self.training:
                ae_logits = ae_logits_mask * ae_logits
            print("ae_logits 's size is {}".format(
                ae_logits.size())) if consts.ONECHANCE else None
            loss_ae, predicted_events = self.calculate_loss_ae(ae_logits, ae_logits_key, batch_golden_events, BATCH_SIZE, trigger_label_s2i)
        else:
            loss_ae = zero_loss
            predicted_events = zero_events
        return loss_ae, predicted_events

    def trigger_entity_combi(self, trigger_type_list, trigger_type_emb, entity_type_list, entity_type_emb):
        def no_duplicate_emb(type_list, type_emb):
            res_type, res_emb = [], []
            for idx in range(len(type_list)):
                if type_list[idx] in type_list[idx + 1:]:
                    continue
                res_emb.append(type_emb[idx])
                res_type.append(type_list[idx])
            return res_type, res_emb
        trigger_type, trigger_emb = no_duplicate_emb(trigger_type_list, trigger_type_emb)
        entity_type, entity_emb = no_duplicate_emb(entity_type_list, entity_type_emb)

        type_cat = [(t_type, e_type) for e_type in entity_type for t_type in trigger_type]
        emb_cat = [(t_emb, e_emb) for e_emb in entity_emb for t_emb in trigger_emb]
        return type_cat, emb_cat

    def add_ae_repr(self, pair_repr, trigger_type_emb, entity_type_emb, ae_input):
        if self.hyperparams["arg_gate"]:
            pair_repr_ = self.arg_gate_repr(pair_repr, trigger_type_emb, entity_type_emb)
        else:
            pair_repr_ = torch.cat([pair_repr, trigger_type_emb, entity_type_emb], dim=-1)
            #pair_repr = pair_repr * trigger_type_emb
            #pair_repr_ = torch.cat([pair_repr, entity_type_emb], dim=-1)
        ae_input.append(pair_repr_)

    def output_arg_mask(self):
        if self.update_mask == False: return
        print(self.arg_mask_init.tolist())
        print(self.arg_mask.tolist())
        self.update_mask = False

    def update_arg_mask(self, trigger_idx, arg_idx):
        if self.update_mask == False: return
        self.arg_mask[trigger_idx, arg_idx] = 1.

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
            if len(batch_golden_events[i]) > 0:
                sample_idx.append(kid)
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

        if self.training:
            output_ae = torch.zeros(golden_labels.size(), dtype=torch.int64).cuda()
            total_idx = torch.LongTensor(sample_idx).cuda()
            #print("total_idx size is {}".format(total_idx.size()))
            logits = logits[total_idx]
            golden_labels = golden_labels[total_idx]
            print("training logits size is {}".format(logits.size())) if consts.ONECHANCE else None
        #loss = F.nll_loss(F.log_softmax(logits, dim=1), golden_labels)
        loss = self.loss(logits, golden_labels)

        predicted_events = [{} for _ in range(BATCH_SIZE)]  # 构造词典，保存每个（正确的）trigger候选的分类正确的argument
        if self.training:
            predict_label = torch.max(logits, 1)[1].view(golden_labels.size())
            output_ae[total_idx] = predict_label
            output_ae = output_ae.tolist()
        else:
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
