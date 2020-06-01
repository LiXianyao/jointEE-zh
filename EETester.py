from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from util import consts
from sklearn import metrics
import numpy as np
from collections import defaultdict

class EETester:

    def calculate_report(self, y, y_, label_i2s, transform=True, output=False, calculate_type=None):
        '''
        calculating F1, P, R

        :param y: golden label, list
        :param y_: model output, list
        :param label_i2s: a mapping from int to label-str on model output
        :param transform: enable the transformation using label_i2s on y_
        :return:
        '''
        if transform:
            for i in range(len(y)):
                for j in range(len(y[i])):
                    y[i][j] = label_i2s[y[i][j]]
            for i in range(len(y_)):
                for j in range(len(y_[i])):
                    y_[i][j] = label_i2s[y_[i][j]]
        if output:
            accuracy = round(accuracy_score(y, y_) * 100., 2)
            classification_report_str = ("\n").join(classification_report(y, y_).split('\n')[-3:])
            print("accuracy = {}, classification report for {} is\n{}".format(accuracy, calculate_type, classification_report_str))
        return round(precision_score(y, y_) * 100., 2), \
               round(recall_score(y, y_) * 100., 2), \
               round(f1_score(y, y_) * 100., 2)

    def calculate_sets(self, y, y_, label_i2s, label_s2i, output=False):
        ct, p1, p2 = 0, 0, 0
        trigger_match = 0
        ay, ay_ = [], []
        for sent, sent_ in zip(y, y_):
            for key, value in sent.items():  # 原句子中的每个event
                p1 += len(value)  # event的golden entities数量
                if key not in sent_:  # 事件没有被预测出来
                    continue
                trigger_match += 1
                # matched sentences
                arguments = value
                arguments_ = sent_[key]

                arguments_dict = [(a_start, a_end, a_type) for (a_start, a_end, a_type, e_type) in arguments]
                arguments_dict_ = [(a_start, a_end, a_type) for (a_start, a_end, a_type, e_type) in arguments_]
                same_argument_list = set(arguments_dict) & set(arguments_dict_)
                #print("arguments", arguments_dict) if output else None
                #print("arguments——", arguments_dict_) if output else None
                ct += len(same_argument_list)  # 正确的argument计数

            for key, value in sent_.items():
                p2 += len(value)  # 预测出的事件候选及其实体数
            #print(sent.keys()) if output else None
            #print(sent_.keys()) if output else None

        sk_p, sk_r, sk_f = 0., 0., 0.
        print(u"golden argument number is {}, predict argument number is {}, correct number is :{}".
              format(p1, p2, ct)) if output else None
        if ct == 0 or p1 == 0 or p2 == 0:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            p, r, f1 = round(p * 100., 2), round(r * 100., 2), round(f1 * 100., 2)
        return (p, sk_p), (r, sk_r), (f1, sk_f)

    def calculate_sets_S(self, y, y_, label_i2s, label_s2i, output=False):
        ct, p1, p2 = 0, 0, 0
        exclude_cnt = [0]
        if label_i2s[consts.PADDING_IDX] == consts.PADDING_LABEL:
            exclude_cnt = [0, consts.PADDING_IDX]

        ay, ay_ = [], []
        for sent, sent_ in zip(y, y_):
            sent_total_key = list(set(  list(sent.keys()) + list(sent_.keys())  ))
            sent_y, sent_y_ = [], []
            for key in sent_total_key:
                arguments = sent[key] if key in sent else []
                arguments_ = sent_[key] if key in sent_ else []
                arguments_dict = {(a_start, a_end): a_type for (a_start, a_end, a_type, e_type) in arguments}
                arguments_dict_ = {(a_start, a_end): a_type for (a_start, a_end, a_type, e_type) in arguments_}
                sum_argument_list = list(set( list(arguments_dict.keys()) + list(arguments_dict_.keys()) ))
                e_type_list = []
                e_type_list_ = []
                for e_key in sum_argument_list:
                    a_type = arguments_dict[e_key] if e_key in arguments_dict else 0
                    a_type_ = arguments_dict_[e_key] if e_key in arguments_dict_ else 0
                    p1 += 1 if a_type else 0  # golden 标签
                    p2 += 1 if a_type_ else 0
                    ct += 1 if a_type and a_type == a_type_ else 0
                    e_type_list.append(label_i2s[a_type])
                    e_type_list_.append(label_i2s[a_type_])
                    sent_y.append(a_type)
                    sent_y_.append(a_type_)
            ay.extend(sent_y)
            ay_.extend(sent_y_)

        sk_p, sk_r, sk_f = 0., 0., 0.
        if output:
            print(u"golden argument number is {}, predict argument number is {}, correct number is :{}".
                  format(p1, p2, ct))
            sk_p, sk_r, sk_f = self._identification(ay, ay_, exclude_cnt)
            print("arg-det p: {} r: {} f1: {} ---Sklearn metrics".format(sk_p, sk_r, sk_f))
            sk_p, sk_r, sk_f = self._classification(ay, ay_, label_i2s)

        if ct == 0 or p1 == 0 or p2 == 0:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            p, r, f1 = round(p * 100., 2), round(r * 100., 2), round(f1 * 100., 2)
        return (p, sk_p), (r, sk_r), (f1, sk_f)

    def calculate_anchors(self, y, y_, calculate_type=None, output=False, use_vocab=False, label_i2s=None):  # 每个anchor矩阵都是 batch * seq * anchor_num
        ct, p1, p2 = defaultdict(int), defaultdict(int), defaultdict(int)
        ay, ay_ = [], []
        exclude_cnt = [0]
        padding_part = [-1]
        if use_vocab and label_i2s[consts.PADDING_IDX] == consts.PADDING_LABEL:
            exclude_cnt = [0, consts.PADDING_IDX]
            padding_part.append(consts.PADDING_IDX)

        for sent, sent_ in zip(y, y_):
            for word, word_ in zip(sent, sent_): # 每个中心词的anchors结果
                ay.extend(word)
                ay_.extend(word_)
                anchor_idx = 0
                for anchor, anchor_ in zip(word, word_):
                    anchor_idx += 1
                    if anchor not in exclude_cnt:
                        p1["sum"] += 1  # ground truth 的一个anchor
                        p1[anchor_idx] += 1
                    if anchor in padding_part:  # padding part
                        continue
                    if anchor_ not in exclude_cnt:  # 预测的一个anchor
                        p2["sum"] += 1
                        p2[anchor_idx] += 1
                    if anchor == anchor_ and anchor not in exclude_cnt:
                        ct["sum"] += 1  # 预测正确的anchor计数
                        ct[anchor_idx] += 1

        if output:
            print(u"golden {}number is {}, predict {} number is {}, correct number is :{}".format(calculate_type, p1["sum"],
                                                                                                  calculate_type, p2["sum"],
                                                                                                  ct["sum"]))
            print(u"correct number for each anchor is: {}".format(ct))
            if use_vocab:
                sk_p, sk_r, sk_f = self._classification(ay, ay_, label_i2s)
            else:
                sk_p, sk_r, sk_f = self._identification(ay, ay_, exclude_cnt)
        else:
            sk_p, sk_r, sk_f =0., 0., 0.

        if ct["sum"] == 0 or p1["sum"] == 0 or p2["sum"] == 0:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct["sum"] / p2["sum"]  # 正确数/预测数 = 准确率
            r = 1.0 * ct["sum"] / p1["sum"]  # 正确数 / 实际正确数 = 召回率
            f1 = 2.0 * p * r / (p + r)
            p, r, f1 = round(p * 100., 2), round(r * 100., 2), round(f1 * 100., 2)
        return (p, sk_p), (r, sk_r), (f1, sk_f)

    def binarize_label(self, labels, exclude_list):
        #print(labels)
        #print("len for identification is", len(labels))
        np_labels = np.array(labels)
        #print(np_labels)
        for exclude in exclude_list:
            np_labels[np_labels == exclude] = 0
        np_labels[np_labels > 0] = 1
        #        np_labels[np_labels > 0] = 1

        return np_labels.tolist()

    def _identification(self, y_true, y_pred, exclude_list):
        if len(y_true) == 0:
            return 0, 0, 0

        if len(set(y_true)) == 1 and y_true[0] == 0:
            return 0, 0, 0
        y_true = self.binarize_label(y_true, exclude_list)
        y_pred = self.binarize_label(y_pred, exclude_list)
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

        return round(p * 100., 2), round(r * 100., 2), round(f1 * 100., 2)

    def _classification(self, y_true, y_pred, label_i2s=None, exclude_other=True, output_dict=True):
        if len(y_true) == 0:
            return 0, 0, 0

        if len(set(y_true)) == 1 and y_true[0] == 0:
            return 0, 0, 0
        # print('classification...')
        # print(y_true)
        # print(y_pred)
        labels = None
        if label_i2s:
            labels = [i for i in range(len(label_i2s))]
            #     print(label_i2s[1])
            if exclude_other:
                exclude_idx = 2 if label_i2s[consts.PADDING_IDX] == consts.PADDING_LABEL else 1
                labels = labels[exclude_idx:]  # label_i2s[1] = 'O'; label_i2s[1] = '<pad>'
                #        labels = labels[1:-1] # label_i2s[1] = 'O'; label_i2s[1] = '<pad>'
                label_i2s = label_i2s[exclude_idx:]
        #        print(labels)
        #        print(label_i2s)
        #print(y_true)
        #print(y_pred)
        #print(labels)
        #print(label_i2s)
        report = metrics.classification_report(y_true, y_pred, digits=2,
                                       labels=labels, target_names=label_i2s, output_dict=output_dict)

        if output_dict:
            p = report["weighted avg"]["precision"]
            r = report["weighted avg"]["recall"]
            f = report["weighted avg"]["f1-score"]
            #print(p, r, f)
            #exit(0)
            return round(p * 100., 2), round(r * 100., 2), round(f * 100., 2)

        return report