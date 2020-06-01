#-*-encoding:utf-8-*-#
import torch
from util.util import *
from util.corpus.Sentence import Token, Anchor
import json

class EpochRunner:
    def __init__(self, word_i2s, EERuner, optimizer_constructor, bert_optimizer_constructor):
        self.word_i2s = word_i2s
        self.EERuner = EERuner
        self.bert_lr = EERuner.arg.bert_lr
        self.bert_rho = EERuner.arg.bert_rho

        self.lr = EERuner.arg.lr
        self.lr_rho = EERuner.arg.lr_rho
        self.triggerSeq_i2s = EERuner.TriggerLabelField.vocab.itos
        #print("trigger_i2s is {}, len = {}".format(self.triggerSeq_i2s, len(self.triggerSeq_i2s)))
        self.triggerAnchor_i2s = EERuner.TriggerAnchorTypesField.vocab.itos
        self.triggerAnchor_s2i = EERuner.TriggerAnchorTypesField.vocab.stoi
        self.entityAnchor_i2s = EERuner.EntityAnchorTypesField.vocab.itos
        self.argument_i2s = EERuner.EventsField.vocab.itos
        self.argument_s2i = EERuner.EventsField.vocab.stoi
        print(self.argument_i2s)
        print(self.argument_s2i)
        self.optimizer_constructor = optimizer_constructor
        self.bert_optimizer_constructor = bert_optimizer_constructor

    def lrdecay(self, epoch_num, lr, rho, constructor, prefix=""):
        lr = lr / (1 + rho * epoch_num)
        print("{}lr of epoch {} is {}".format(prefix, epoch_num, lr))
        return constructor(lr=lr)

    def warm_up_lrdecay(self, epoch_num, lr, rho, constructor, hyps, prefix=""):
        lr = lr / (1 + rho * epoch_num)
        if epoch_num >= hyps["warmup_epoch"]:
            lr *= hyps["warmup"]
        print("{}lr of epoch {} is {}".format(prefix, epoch_num, lr))
        return constructor(lr=lr)

    def run_one_epoch(self, model, data_iter, MAX_STEP, epoch_num, need_backward, tester, hyps, device, save_output, maxnorm):
        optimizer = self.warm_up_lrdecay(epoch_num=epoch_num, lr=self.lr, rho=self.lr_rho,
                                         constructor=self.optimizer_constructor, hyps=hyps)
        bert_optimizer = self.warm_up_lrdecay(epoch_num=epoch_num, lr=self.bert_lr, rho=self.bert_rho,
                                         constructor=self.bert_optimizer_constructor, hyps=hyps, prefix="bert_")
        #optimizer, bert_optimizer = self.lrdecay(epoch_num=epoch_num)
        if need_backward:
            model.test_mode_off()
        else:
            model.test_mode_on()

        ed_loss = 0.0
        edc_loss = 0.0
        emd_loss = 0.0
        emdc_loss = 0.0
        arg_loss = 0.0

        all_trigger_anchor_label = []
        all_trigger_anchor_label_ = []
        all_trigger_anchor_cls = []
        all_trigger_anchor_cls_ = []
        all_trigger_anchor = []
        all_trigger_cls = []

        all_entity_anchor_label = []
        all_entity_anchor_label_ = []
        all_entity_anchor_cls = []
        all_entity_anchor_cls_ = []
        all_entity_anchor = []
        all_entity_cls = []

        all_events_arg = []
        all_events_arg_ = []
        output_event_json = []
        output_trigger_weight_json = []
        batch_cnt = 0
        for batch in data_iter:
            if need_backward:
                optimizer.zero_grad()
                bert_optimizer.zero_grad()

            words, w_len = batch.WORDS  # (batch, seqlen), (batch)
            postags = batch.POSTAGS
            #entity_bio_y = batch.ENTITYLABELS
            #entity_anchors_cls = batch.ENTITYANCHORCLS  # (batch, seqlen, 3)
            # print("ANCHORCLS size is{}, contents are:\n{}".format(anchors_cls.size(), anchors_cls))
            #entity_anchors_label = batch.ENTITYANCHORLABEL
            #entity_anchor = batch.ENTITYANCHOR
            tokens, t_len = batch.TOKENS  # (batch, seqlen), (batch)

            token_segments = torch.zeros(tokens.size(), dtype=torch.long)
            word_segments = torch.zeros(postags.size(), dtype=torch.long)
            tokens_mask = batch.TOKENSMASK  # [batch, maxlen, maxlen]
            #print("tokens_mask size is{}, contents are:\n{}".format(tokens_mask.size(), tokens_mask))

            trigger_seq_tag, trigger_seq_len = batch.TRIGGERLABEL
            trigger_anchor_cls = batch.TRIGGERANCHORCLS  # (batch, seqlen, 3)
            # print("ANCHORCLS size is{}, contents are:\n{}".format(anchors_cls.size(), anchors_cls))
            trigger_anchor_label = batch.TRIGGERANCHORLABEL
            trigger_anchor = batch.TRIGGERANCHOR

            entity_anchors_cls = batch.ENTITYANCHORCLS  # (batch, seqlen, 3)
            # print("ANCHORCLS size is{}, contents are:\n{}".format(anchors_cls.size(), anchors_cls))
            entity_anchors_label = batch.ENTITYANCHORLABEL
            entity_anchor = batch.ENTITYANCHOR

            event = batch.EVENTS  # json{   }
            all_events_arg.extend(event)

            if device.type != "cpu":
                tokens = tokens.cuda(device.index)
                postags = postags.cuda(device.index)
                token_segments = token_segments.cuda(device.index)
                word_segments = word_segments.cuda(device.index)
                t_len = t_len.cuda(device.index)
                tokens_mask = tokens_mask.cuda(device.index)
                trigger_seq_tag = trigger_seq_tag.cuda(device.index)
                trigger_seq_len = trigger_seq_len.cuda(device.index)
                trigger_anchor = trigger_anchor.cuda(device.index)
                trigger_anchor_cls = trigger_anchor_cls.cuda(device.index)
                trigger_anchor_label = trigger_anchor_label.cuda(device.index)
                entity_anchor = entity_anchor.cuda(device.index)
                entity_anchors_cls = entity_anchors_cls.cuda(device.index)
                entity_anchors_label = entity_anchors_label.cuda(device.index)
            # batch, seqlen, anchor_num, 2
            loss_ed, ed_detect_label, loss_ed_cls, ed_cls_label, loss_emd, emd_det_label, loss_emd_cls, emd_cls_label, loss_ae, predicted_events\
                , trigger_candidates_weight = model.forward(
                    input_ids=tokens, input_len=t_len, segments=(token_segments, word_segments),
                    trigger_labels=trigger_seq_tag, token_head_mask=tokens_mask, postags=postags,

                    trigger_labels_len=trigger_seq_len, trigger_anchor_loc=trigger_anchor,
                    trigger_anchor_labels=trigger_anchor_label, trigger_anchor_type=trigger_anchor_cls,

                    entity_anchor_loc=entity_anchor, entity_anchor_labels=entity_anchors_label,
                    entity_anchor_type=entity_anchors_cls,

                    trigger_label_i2s=self.triggerAnchor_i2s, trigger_label_s2i=self.triggerAnchor_s2i, entity_label_i2s=self.entityAnchor_i2s,
                    batch_golden_events=event)

            bp, br, bf = self.add_anchors(words, trigger_anchor_label, ed_detect_label, w_len, all_trigger_anchor, self.word_i2s,
                                          tester, all_trigger_anchor_label, all_trigger_anchor_label_, save_output=save_output)  # 单个batch的评价结果
            if hyps["ed_cls_mode"] != "crf":
                bcp, bcr, bcf = self.add_anchors(words, trigger_anchor_cls, ed_cls_label, w_len, all_trigger_cls,
                                                 self.word_i2s, tester, all_trigger_anchor_cls, all_trigger_anchor_cls_,
                                                 label_i2s=self.triggerAnchor_i2s, use_vocab=True, save_output=save_output)  # 单个batch的评价结果
            else:
                bcp, bcr, bcf = self.add_tokens(words, trigger_seq_tag, ed_cls_label, trigger_seq_len, all_trigger_cls,
                                                 self.word_i2s, tester, all_trigger_anchor_cls, all_trigger_anchor_cls_,
                                                 label_i2s=self.triggerSeq_i2s, use_vocab=True)  # 单个batch的评价结果
            ep, er, ef = self.add_anchors(words, entity_anchors_label, emd_det_label, w_len, all_entity_anchor, self.word_i2s,
                                          tester, all_entity_anchor_label, all_entity_anchor_label_, save_output=save_output)  # 单个batch的评价结果
            ecp, ecr, ecf = self.add_anchors(words, entity_anchors_cls, emd_cls_label, w_len, all_entity_cls, self.word_i2s,
                                             tester, all_entity_anchor_cls, all_entity_anchor_cls_,
                                             label_i2s=self.entityAnchor_i2s, use_vocab=True, save_output=save_output)  # 单个batch的评价结果
            ap, ar, af = tester.calculate_sets_S(event, predicted_events, self.argument_i2s, self.argument_s2i) if save_output else 0, 0, 0

            loss = torch.zeros([1]).cuda(device)
            if hyps["ED_enable"]:
                loss += hyps["alpha_ed"] * loss_ed + hyps["alpha_edc"] * loss_ed_cls
                ed_loss += hyps["alpha_ed"] * loss_ed.item()
                edc_loss += hyps["alpha_edc"] * loss_ed_cls.item()
                if save_output:
                    trigger_att_json = self.output_trigger_jsons(words, w_len, trigger_candidates_weight)
                    output_trigger_weight_json.extend(trigger_att_json)
            if hyps["EMD_enable"]:
                loss += hyps["beta_emd"] * loss_emd + hyps["beta_emd"] * loss_emd_cls
                emd_loss += hyps["beta_emd"] * loss_emd.item()
                emdc_loss += hyps["beta_emd"] * loss_emd_cls.item()
            if hyps["ARG_enable"]:
                loss += hyps["gama_arg"] * loss_ae
                arg_loss += hyps["gama_arg"] * loss_ae.item()
                all_events_arg_.extend(predicted_events)
                if save_output:
                    event_json = self.output_event_jsons(words, w_len, event, predicted_events)
                    output_event_json.extend(event_json)

            batch_cnt += 1
            if need_backward:
                loss.backward()
                if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:  # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)
                optimizer.step()
                bert_optimizer.step()

            if batch_cnt % max(MAX_STEP // 5, 1) == 0:
                other_information = 'Iter[{}] loss: {:.6f}'.\
                    format(batch_cnt, loss.item())
                progressbar(batch_cnt, MAX_STEP, other_information)

        """epoch finished, save output if necessary"""
        if save_output:
            self.save_token_output(output_path=save_output, type="trigger_anchor", all_anchor=all_trigger_anchor)
            self.save_token_output(output_path=save_output, type="trigger_cls", all_anchor=all_trigger_cls)
            self.save_token_output(output_path=save_output, type="entity_anchor", all_anchor=all_entity_anchor)
            self.save_json_output(output_path=save_output, type="argument_role", json_data=output_event_json)
            self.save_json_output(output_path=save_output, type="trigger_att_weight", json_data=output_trigger_weight_json)
            self.save_token_output(output_path=save_output, type="entity_cls", all_anchor=all_entity_cls)


        ed_loss /= batch_cnt
        edc_loss /= batch_cnt
        emdc_loss /= batch_cnt
        emd_loss /= batch_cnt
        arg_loss /= batch_cnt

        edp, edr, edf = tester.calculate_anchors(all_trigger_anchor_label, all_trigger_anchor_label_,calculate_type="trigger", output=True)
        if hyps["ed_cls_mode"] != "crf":
            edcp, edcr, edcf = tester.calculate_anchors(all_trigger_anchor_cls, all_trigger_anchor_cls_,
                                                        calculate_type="trigger", output=True, use_vocab=True, label_i2s=self.triggerAnchor_i2s)
        else:
            edcp, edcr, edcf = tester.calculate_report(all_trigger_anchor_cls, all_trigger_anchor_cls_,
                                                       self.triggerSeq_i2s, transform=False, calculate_type="trigger",
                                                       output=True)
        emdp, emdr, emdf = tester.calculate_anchors(all_entity_anchor_label, all_entity_anchor_label_,
                                                    output=True, calculate_type="entity")
        emdcp, emdcr, emdcf = tester.calculate_anchors(all_entity_anchor_cls, all_entity_anchor_cls_, output=True,
                                                    use_vocab=True, calculate_type="entity", label_i2s=self.entityAnchor_i2s)
        argp, argr, argf = tester.calculate_sets_S(all_events_arg, all_events_arg_, self.argument_i2s, self.argument_s2i, output=True)
        return ed_loss, edc_loss, emd_loss, emdc_loss, arg_loss,\
               edp, edr, edf, edcp, edcr, edcf, \
               emdp, emdr, emdf, emdcp, emdcr, emdcf, \
               argp, argr, argf

    def save_token_output(self, output_path, type, all_anchor):
        with open("{}_{}.txt".format(output_path, type), "w", encoding="utf-8") as f:
            for anchors in all_anchor:
                for anchor in anchors:
                    f.write("{} {}\n".format(anchor.word, anchor.AnchorRes))
                f.write("\n")

    def save_json_output(self, output_path, type, json_data):
        with open("{}_{}.json".format(output_path, type), "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    def set_device(self, device, words):
        pass

    def add_tokens(self, words, tokens_label, predicted_label, w_len, output_tokens, word_i2s, tester,
                    token_label_list, predicted_label_list, label_i2s=None, use_vocab=False):
        words = words.tolist()
        predicted_label = predicted_label  #.tolist()  # batch, seqlen
        tokens_label = tokens_label.tolist()  # batch, seqlen

        for sen, sen_label, sen_predicted, sen_len in zip(words, tokens_label, predicted_label, w_len):
            sen = sen[1:sen_len - 1]
            sen_label = sen_label[1:sen_len - 1]
            sen_predicted = sen_predicted[1:sen_len - 1]
            tokens = []
            for word, word_label, word_predicted in zip(sen, sen_label, sen_predicted):
                if label_i2s:
                    word_label = label_i2s[word_label]
                    word_predicted = label_i2s[word_predicted]
                atoken = Anchor(word=word_i2s[word], posLabel="", AnchorLabel=word_label,
                                predictedAnchorLabel=word_predicted, anchor=False)
                tokens.append(atoken)
            output_tokens.append(tokens)

        for i, length in enumerate(w_len):
            print(words[i])
            tokens_label[i] = tokens_label[i][1:length - 1]
            print(length, len(predicted_label[i]))
            print(tokens_label[i])
            predicted_label[i] = predicted_label[i][1:length - 1]
        bp, br, bf = tester.calculate_report(tokens_label, predicted_label, label_i2s, transform=True)  # 单个batch的评价结果
        token_label_list.extend(tokens_label)
        predicted_label_list.extend(predicted_label)
        return bp, br, bf

    def add_anchors(self, words, anchors_label, predicted_label, w_len, output_anchors, word_i2s, tester,
                    anchor_label_list, predicted_label_list, label_i2s=None, use_vocab=False, save_output=None):
        words = words.tolist()
        predicted_label = predicted_label.tolist()  # batch, seqlen, anchor_num
        anchors_label = anchors_label.tolist()  # batch, seqlen, anchor_num

        for i, length in enumerate(w_len):
            anchors_label[i] = anchors_label[i][:length]
            predicted_label[i] = predicted_label[i][:length]
        anchor_label_list.extend(anchors_label)
        predicted_label_list.extend(predicted_label)

        if not save_output:
            return 0., 0., 0.

        for sen, sen_label, sen_predicted, sen_len in zip(words, anchors_label, predicted_label, w_len):
            sen = sen[:sen_len]
            sen_label = sen_label[:sen_len]
            sen_predicted = sen_predicted[:sen_len]
            anchors = []
            for word, word_label, word_predicted in zip(sen, sen_label, sen_predicted):
                if label_i2s:
                    word_label = [label_i2s[label] for label in word_label]
                    word_predicted = [label_i2s[label] for label in word_predicted]
                atoken = Anchor(word=word_i2s[word], posLabel="", AnchorLabel=word_label,
                                predictedAnchorLabel=word_predicted)
                anchors.append(atoken)
            output_anchors.append(anchors)
        bp, br, bf = tester.calculate_anchors(anchors_label, predicted_label, use_vocab=use_vocab, label_i2s=label_i2s)  # 单个batch的评价结果
        return bp[0], br[0], bf[0]

    def output_event_jsons(self, words, word_len, events, predict_events):
        batch_size = words.size()[0]
        sentenses_text = self.EERuner.recover_data(words, word_len, self.EERuner.WordsField, depth=2)  # list for batch-sentences
        event_jsons = []
        for b_idx in range(batch_size):
            sentense_dict = {"text": sentenses_text[b_idx], "event":[]}
            sentense_word = sentenses_text[b_idx].split(" ")[1:]

            trigger_dict = {(t_start, t_end): t_type for (t_start, t_end, t_type) in events[b_idx]}
            trigger_dict_ = {(t_start, t_end): t_type for (t_start, t_end, t_type) in predict_events[b_idx]}
            all_trigger_idx = set(trigger_dict) | set(trigger_dict_)

            for start, end in all_trigger_idx:
                event_type = trigger_dict[(start, end)] if (start, end) in trigger_dict else "OTHER"
                predict_event_type = trigger_dict_[(start, end)] if (start, end) in trigger_dict_ else "OTHER"

                arguments = events[b_idx][(start, end, event_type)] if event_type != "OTHER" else []
                arguments_ = predict_events[b_idx][(start, end, predict_event_type)] \
                    if predict_event_type != "OTHER" else []

                arguments_dict = {(a_start, a_end): (a_type, e_type) for (a_start, a_end, a_type, e_type) in arguments}
                arguments_dict_ = {(a_start, a_end): (a_type, e_type) for (a_start, a_end, a_type, e_type) in arguments_}
                all_argument_idx = set(arguments_dict) | set(arguments_dict_)

                argument_list = []
                for (a_start, a_end) in all_argument_idx:
                    role = self.argument_i2s[arguments_dict[(a_start, a_end)][0]] \
                        if (a_start, a_end) in arguments_dict else "OTHER"
                    predict_role = self.argument_i2s[arguments_dict_[(a_start, a_end)][0]] \
                        if (a_start, a_end) in arguments_dict_ else "OTHER"
                    entity_type = arguments_dict[(a_start, a_end)][1] if (a_start, a_end) in arguments_dict else "OTHER"
                    predict_entity_type = arguments_dict_[(a_start, a_end)][1] if (a_start, a_end) in arguments_dict_ else "OTHER"
                    argument = {
                        "role": role,
                        "predict_role": predict_role,
                        "entity_type": entity_type,
                        "predict_entity": predict_entity_type,
                        "loc": [a_start, a_end],
                        "text": sentense_word[a_start: a_end],
                        "arg_correct": role == predict_role,
                        "e_correct": entity_type == predict_entity_type
                    }
                    argument_list.append(argument)
                trigger_unit = {
                    "trigger": {
                        "loc": [start, end],
                        "text": sentense_word[start: end],
                        "event_type": event_type,
                        "predict_event_type": predict_event_type,
                        "event_correct": event_type == predict_event_type
                    },
                    "argument": argument_list
                }
                sentense_dict["event"].append(trigger_unit)
            event_jsons.append(sentense_dict)
        return event_jsons

    def output_trigger_jsons(self, words, word_len, trigger_att_dict):
        batch_size = words.size()[0]
        sentenses_text = self.EERuner.recover_data(words, word_len, self.EERuner.WordsField, depth=2)  # list for batch-sentences
        trigger_jsons = []
        for b_idx in range(batch_size):
            sentense_dict = {"text": sentenses_text[b_idx], "trigger":[]}
            sentense_word = sentenses_text[b_idx].split(" ")[1:]

            for start, end in trigger_att_dict[b_idx]:
                all_candidates = trigger_att_dict[b_idx][(start, end)]["keys"]
                candidates_list = []
                for (a_start, a_end), weight in all_candidates:
                    candidate = {
                        "loc": [a_start, a_end],
                        "text": sentense_word[a_start: a_end],
                        "weight": weight
                    }
                    candidates_list.append(candidate)
                trigger_unit = {
                    "trigger": {
                        "loc": [start, end],
                        "text": sentense_word[start: end],
                        "label": self.triggerAnchor_i2s[trigger_att_dict[b_idx][(start, end)]["label"]],
                        "predict": self.triggerAnchor_i2s[trigger_att_dict[b_idx][(start, end)]["predict"]],
                        "t_correct": trigger_att_dict[b_idx][(start, end)]["t_correct"]
                    },
                    "candidates": sorted(candidates_list, key=lambda candidate: candidate['weight'], reverse=True)
                }
                sentense_dict["trigger"].append(trigger_unit)
            trigger_jsons.append(sentense_dict)
        return trigger_jsons
