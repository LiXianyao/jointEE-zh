import json
from torchtext.data import Example

from util.corpus.Corpus import Corpus
from util.corpus.Sentence import Sentence


class ACE2005Dataset(Corpus):
    """
    Defines a dataset composed of Examples along with its Fields.
    """

    sort_key = None

    def __init__(self, path, tokenizer, fields, keep_events=None, only_keep=False, skip_sample=None, **kwargs):
        '''
        Create a corpus given a path, field list, and a filter function.

        :param path: str, Path to the data file
        :param fields: dict[str: tuple(str, Field)],
                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
        :param keep_events: int, minimum sentence events. Default keep all.
        '''
        self.keep_events = keep_events
        self.only_keep = only_keep
        self.skip_sample = skip_sample
        self.tokenizer = tokenizer
        super(ACE2005Dataset, self).__init__(path, fields, **kwargs)

    def parse_example(self, path, fields):
        examples = []
        false_cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            jl = json.load(f)
            for js in jl:
                keep, ex = self.parse_sentence(js, fields)
                if not keep:
                    false_cnt += 1
                    if (self.skip_sample is None) or (false_cnt % self.skip_sample):  # for instance, if skip_sample=2, than roughly half of the data unsatisfied would be skip
                        continue
                if ex is not None:
                    examples.append(ex)

        return examples

    def parse_sentence(self, js, fields):
        WORDS = fields["words"]
        TOKENS = fields["tokens"]
        POSTAGS = fields["pos-tags"]
        TOKENSMASK = fields["tokens_mask"]
        #ENTITYLABELS = fields["golden-entity-mentions"]
        ENTITIYANCHOR = fields["entity-anchor"]
        ENTITIYANCHORLABEL = fields["entity-anchor-label"]
        ENTITIYANCHORCLS = fields["entity-anchor-class"]
        #ADJMATRIX = fields["stanford-colcc"]
        TRIGGERLABEL = fields["golden-event-mentions"]
        TRIGGERANCHOR = fields["trigger-anchor"]
        TRIGGERANCHORLABEL = fields["trigger-anchor-label"]
        TRIGGERANCHORCLS = fields["trigger-anchor-class"]
        EVENTS = fields["all-events"]
        #ENTITIES = fields["all-entities"]

        sentence = Sentence(json_content=js, tokenizer=self.tokenizer)
        ex = Example()
        setattr(ex, WORDS[0], WORDS[1].preprocess(sentence.wordList))
        setattr(ex, TOKENS[0], TOKENS[1].preprocess(sentence.tokens))
        setattr(ex, POSTAGS[0], POSTAGS[1].preprocess(sentence.posLabelList))
        setattr(ex, TOKENSMASK[0], TOKENSMASK[1].preprocess(sentence.token_valid))
        #setattr(ex, ENTITYLABELS[0], ENTITYLABELS[1].preprocess(sentence.entityLabelList))
        setattr(ex, ENTITIYANCHOR[0], ENTITIYANCHOR[1].preprocess(sentence.entityAnchorList))
        setattr(ex, ENTITIYANCHORLABEL[0], ENTITIYANCHORLABEL[1].preprocess(sentence.entityAnchorLableList))
        setattr(ex, ENTITIYANCHORCLS[0], ENTITIYANCHORCLS[1].preprocess(sentence.entityAnchorClassList))
        setattr(ex, TRIGGERLABEL[0], TRIGGERLABEL[1].preprocess(sentence.triggerLabelList))
        setattr(ex, TRIGGERANCHOR[0], TRIGGERANCHOR[1].preprocess(sentence.triggerAnchorList))
        setattr(ex, TRIGGERANCHORLABEL[0], TRIGGERANCHORLABEL[1].preprocess(sentence.triggerAnchorLableList))
        setattr(ex, TRIGGERANCHORCLS[0], TRIGGERANCHORCLS[1].preprocess(sentence.triggerAnchorClassList))
        setattr(ex, EVENTS[0], EVENTS[1].preprocess(sentence.events))
        #setattr(ex, ENTITIES[0], ENTITIES[1].preprocess(sentence.entities))
        #print(ex.__dict__)
        if self.keep_events is not None:
            if self.only_keep and sentence.containsEvents != self.keep_events:  # 只保留数量为keep events的数据
                return False, None
            elif not self.only_keep and sentence.containsEvents < self.keep_events:  # 保留数量大于等于keep events的数据
                return False, ex
            else:
                return True, ex
        else:
            return True, ex

    def longest(self):
        return max([len(x.TOKENS) for x in self.examples])
