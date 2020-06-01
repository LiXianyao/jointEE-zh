from util import consts
import numpy as np

def is_number(s):
    try:
        float(s)  # for int, long and float
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False

    return True

def pretty_str(a):
    a = a.upper()
    if a == 'O':
        return a
    elif a[1] == '-':
        return a[:2] + "|".join(a[2:].split("-")).replace(":", "||")
    else:
        return "|".join(a.split("-")).replace(":", "||")


class Sentence:
    """process data of a sentence into required data"""

    def __init__(self, json_content, tokenizer):
        ## word and char inputs
        self.tokenizer = tokenizer
        self.wordList = json_content["words"][:consts.CUTOFF - 2]  # senlen
        self.tokens, self.token_valid, token_word_len = self.generateTokenList()  # , self.segment_ids = self.generateTokenList()
        self.word_length = len(self.wordList)
        assert token_word_len == self.word_length
        self.sen_length = len(self.tokens)


        # for every word in the sentence, generate k entity candidates for each word, so as the anchors'label and classType
        self.entityLabelList, entity_pos_dict = self.generateEntityLabelList(json_content["golden-entity-mentions"])  # senlen * overlapping_entities_num
        self.entityAnchorList, self.entityAnchorLableList, self.entityAnchorClassList, entity_anchor_map = self.\
            generateCandidatesLabelList(entity_pos_dict, k=consts.ENTITY_ANCHOR_NUM)

        #for every word in the sentence, generate 3 trigger candidates for each word, so as the anchors'label and classType
        self.triggerLabelList, trigger_pos_dict = self.generateTriggerLabelList(json_content["golden-event-mentions"])
        self.triggerAnchorList, self.triggerAnchorLableList, self.triggerAnchorClassList, trigger_anchor_map =\
            self.generateCandidatesLabelList(trigger_pos_dict, k=consts.TRIGGER_ANCHOR_NUM)
        # the length of triggers' distribution:  {1: (4124, 95.37%), 2: (174, 4.02%), 3: (25, 0.58%), 7: (1, 0.02%)}


        #self.adjpos, self.adjv = self.generateAdjMatrix(json_content["stanford-colcc"])

        #self.entities = self.generateGoldenEntities(json_content["golden-entity-mentions"])
        self.events = self.generateGoldenEvents(json_content["golden-event-mentions"], entity_pos_dict, trigger_pos_dict)

        self.containsEvents = len(json_content["golden-event-mentions"])
        self.wordList = ["[CLS]"] + self.wordList[:token_word_len] + ["[SEP]"]
        self.posLabelList = ["[CLS]"] + json_content["pos-tags"][:token_word_len] + ["[SEP]"]

    def generateTokenList(self):
        valid = []
        tokens = []
        tokens_loc = []
        for i, word in enumerate(self.wordList):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            word_valid = [0] * len(token)
            word_valid[0] = 1  # only keep the first token of a word
            valid.extend(word_valid)
            tokens_loc.extend([i + 1] * len(token))
        #print("word len = {}, with token len = {}".format(len(self.wordList), len(tokens)))
        # keep the place for CLS and SEP
        token_word_len = valid.count(1)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        tokens_loc = [0] + tokens_loc + [token_word_len + 1]
        valid = [1] + valid + [1]
        valid_matrix = self.generateToken2WordProbMatrix(token_word_len, tokens_loc, valid, type=consts.TOKEN_MASK_TYPE)
        #segment_ids = [0] * len(tokens)
        return tokens, valid_matrix, token_word_len

    def generateToken2WordProbMatrix(self, token_word_len, tokens_loc, valid, type='average'):
        start = 0
        matrix = np.zeros((len(valid), token_word_len + 2), dtype=np.float)
        for i in range(token_word_len + 2):
            token_len = tokens_loc.count(i)
            if type == 'average' and token_len > 1:
                matrix[start + 1 : start + token_len, i] = 1./ (2 * (token_len - 1) )
                matrix[start, i] = 0.5
            else:  # type=='first'
                matrix[start, i] = 1.
            start += token_len
        return matrix.tolist()

    def generateCandidatesLabelList(self, golden_pos_dict, k=4):
        '''
        Generate anchors and their labels、type， so that the model can predict the overlapping entity/trigger labels with multiple words
        :param entitiesJsonList:
        :return:
        '''
        loc_map = {}
        def get_anchor_label(idx1, idx2, golden_unit, sen_len):
            """
            Return the label of an input idx pair
            1, id: positive pair, class_id
            0, 0: negitive pair
            -1, -1: unvalid pair
            """
            if idx1 >= 0 and idx2 <= sen_len:
                if idx1 in golden_unit:  # 起点在这个dict里
                    for candidate in golden_unit[idx1]:  # 找到有相同终点的entity
                        if candidate[0] == idx2:
                            return 1, candidate[1]  # true entity pair
                return 0, "OTHER"  # neg pair
            return -1, "OTHER"  # labels out of boundary

        anchors = []  # sen_len * k * 2
        anchors_label = []  # sen_len * k
        anchors_cls = []  # sen_len * k
        valid_len = self.word_length
        for idx in range(valid_len):
            sid, eid = idx, idx + 1
            sdelta, edelta = 0, 1
            anchors.append([])
            anchors_label.append([])
            anchors_cls.append([])
            for kid in range(k):
                anchor_label, anchor_cls = get_anchor_label(sid, eid, golden_pos_dict, valid_len)
                if anchor_label == 1:
                    loc_map[(sid, eid)] = (idx, kid)
                if anchor_label == -1:
                    anchors[-1].append([-1, -1])  # out of boundary takes -1
                    anchor_label = 0
                else:
                    anchors[-1].append([sid + 1, eid + 1])  # CLS take place of the idx 0
                anchors_label[-1].append(anchor_label)
                anchors_cls[-1].append(anchor_cls)
                sid = sid - sdelta
                eid = eid + edelta
                sdelta, edelta = edelta, sdelta
        #print(anchors, anchors_label, anchors_cls)
        anchors = [ [[0, 0] for _ in range(k)] ] + anchors + [ [[valid_len + 1, valid_len + 1] for _ in range(k)] ]
        anchors_label = [[0] * k] + anchors_label + [[0] * k]
        anchors_cls = [["OTHER"] * k] + anchors_cls + [["OTHER"] * k]
        #print(anchors, anchors_label, anchors_cls)
        #exit(0)
        return anchors, anchors_label, anchors_cls, loc_map

    def generateEntityLabelList(self, entitiesJsonList):
        '''
        Keep the overlapping entity labels
        :param entitiesJsonList:
        :return:
        '''

        entityLabel = [["O"] for _ in range(self.word_length)]
        entity_pos_dict = {}

        def assignEntityLabel(index, label):
            if len(entityLabel[index]) == 1 and entityLabel[index][0] == "O":
                entityLabel[index][0] = pretty_str(label)
            else:
                entityLabel[index].append(pretty_str(label))

        for entityJson in entitiesJsonList:
            start = entityJson["start"]
            end = entityJson["end"]
            if end > self.word_length: continue

            if end - start > consts.ENTITY_ANCHOR_NUM:  # ignore the entities that longer than we predict
                continue

            etype = entityJson["entity-type"].split(":")[0]

            if start not in entity_pos_dict:
                entity_pos_dict[start] = []
            entity_pos_dict[start].append((end, etype))

            assignEntityLabel(start, "B-" + etype)
            for i in range(start + 1, end):
                assignEntityLabel(i, "I-" + etype)
        entityLabel = [["O"]] + entityLabel[: self.word_length] + [["O"]]
        return entityLabel, entity_pos_dict

    def generateGoldenEntities(self, entitiesJson):
        '''
        [(2, 3, "entity_type")]
        '''
        golden_list = []
        for entityJson in entitiesJson:
            start = entityJson["start"]  # [CLS]
            end = entityJson["end"]
            if end > self.word_length:  # [SEP]
                continue

            etype = entityJson["entity-type"].split(":")[0]
            golden_list.append((start, end, etype))
        return golden_list

    def generateGoldenEvents(self, eventsJson, entity_pos_dict, trigger_pos_dict):
        '''

        {
            (2, 3, "event_type_str") --> [(1, 2, "role_type_str"), ...]
            ...
        }
        ++ for bert need to take [CLS] and [SEP] into concern
        '''
        def check_in_list(idx1, idx2, golden_unit):
            if idx1 in golden_unit:  # 起点在这个dict里
                for candidate in golden_unit[idx1]:  # 找到有相同终点的entity
                    if candidate[0] == idx2:
                        return True  # true entity pair
            return False # neg pair

        golden_dict = {}
        for eventJson in eventsJson:
            triggerJson = eventJson["trigger"]
            if triggerJson["end"] > self.word_length:
                continue
            t_start = triggerJson["start"]
            t_end = triggerJson["end"]
            if not check_in_list(t_start, t_end, trigger_pos_dict): continue
            key = (t_start + 1, t_end + 1, eventJson["event_type"])
            values = []

            for argumentJson in eventJson["arguments"]:
                if argumentJson["end"] > self.word_length:
                    continue
                role = argumentJson["role"].split("-")[0]
                entity_type = argumentJson["entity-type"].split(":")[0]
                a_start = argumentJson["start"]
                a_end = argumentJson["end"]
                if not check_in_list(a_start, a_end, entity_pos_dict): continue
                consts.TRIGGER_ARG_MAP[eventJson["event_type"]].add(pretty_str(role))
                value = (a_start + 1, a_end + 1, pretty_str(role), entity_type)
                values.append(value)
            golden_dict[key] = list(sorted(values))
        return golden_dict

    def generateTriggerLabelList(self, triggerJsonList):
        triggerLabel = ["O" for _ in range(self.word_length)]
        trigger_pos_dict = {}

        def assignTriggerLabel(index, label):  # 因为trigger没有Overlap现象
            triggerLabel[index] = pretty_str(label)

        for eventJson in triggerJsonList:
            triggerJson = eventJson["trigger"]
            start = triggerJson["start"]
            end = triggerJson["end"]
            if end > self.word_length: continue

            if end - start > consts.TRIGGER_ANCHOR_NUM:  # ignore the entities that longer than we predict
                continue

            etype = eventJson["event_type"]  # .split(":")[0]

            if start not in trigger_pos_dict:
                trigger_pos_dict[start] = []
            trigger_pos_dict[start].append((end, etype))

            assignTriggerLabel(start, "B-" + etype)
            for i in range(start + 1, end):
                assignTriggerLabel(i, "I-" + etype)
        #triggerLabel = ["[CLS]"] + triggerLabel[: self.word_length] + ["[SEP]"]
        triggerLabel = ["O"] + triggerLabel + ["O"]
        return triggerLabel, trigger_pos_dict

    def __len__(self):
        return self.word_length


class Token:
    def __init__(self, word, posLabel, lemmaLabel, entityLabel, triggerLabel):
        self.word = word
        self.posLabel = posLabel
        self.lemmaLabel = lemmaLabel
        self.entityLabel = entityLabel
        self.triggerLabel = triggerLabel
        self.predictedLabel = None

    def addPredictedLabel(self, label):
        self.predictedLabel = label


class Anchor:
    def __init__(self, word, posLabel, AnchorLabel, predictedAnchorLabel, anchor=True):
        self.word = word
        self.posLabel = posLabel
        self.AnchorLabel = AnchorLabel  # list of k anchors labels
        self.predictedAnchorLabel = predictedAnchorLabel
        if anchor:
            self.AnchorRes = self.formAnchorRes(self.AnchorLabel, self.predictedAnchorLabel)
        else:
            if self.AnchorLabel == 'O': self.AnchorLabel = '0'
            if predictedAnchorLabel == 'O': predictedAnchorLabel = '0'
            self.AnchorRes = "{} {} {}".format(self.AnchorLabel, predictedAnchorLabel, self.AnchorLabel == predictedAnchorLabel)

    def formAnchorRes(self, anchorLabel, predictedAnchorLabel):
        anchor_num = len(anchorLabel)
        stringfy = " || ".join(["{}:{}={}".format(anchorLabel[idx], predictedAnchorLabel[idx], anchorLabel[idx]==predictedAnchorLabel[idx]) for idx in range(anchor_num)])
        return stringfy
