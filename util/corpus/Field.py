#-*-encoding:utf-8-*-#
import json
from collections import Counter, OrderedDict

import six
import torch
from torchtext.data import Field, Pipeline, Dataset

class SparseField(Field):
    def process(self, batch, device, train):
        return batch


class EntityField(Field):
    '''
    Processing data each sentence has only one

    [(2, 3, "entity_type")]
    '''

    def preprocess(self, x):
        return x

    def pad(self, minibatch):
        return minibatch

    def numericalize(self, arr, device=None, train=True):
        return arr


class EventField(Field):
    '''
    Processing data each sentence has only one

    {
            (2, 3, "event_type_str") --> [(1, 2, "role_type_str"), ...]
            ...
    }
    '''

    def preprocess(self, x):
        return x

    def build_vocab(self, *args, **kwargs):
        '''
        only construct a diction for role_type_str
        {
            (2, 3, "event_type_str") --> [(1, 2, "role_type_str"), ...]
        }
        '''
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                for key, value in x.items():
                    for v in value:  # start, end, role_type
                        counter.update([v[2]])
        self.vocab = self.vocab_cls(counter, specials=["OTHER"], **kwargs)

    def pad(self, minibatch):
        return minibatch

    def numericalize(self, arr, device=None, train=True):
        if self.use_vocab:
            """ just transform the role_type_str in the json into  role type idx"""
            arr = [{key: [(v[0], v[1], self.vocab.stoi[v[2]], v[3]) for v in value] for key, value in dd.items()} for dd in
                   arr]
        return arr


class TokenField(Field):
    '''
    Processing data already tokenize by bert
    '''

    def preprocess(self, x):
        return x

    def build_vocab(self, tokenizer, *args, **kwargs):
        counter = Counter()
        try:
            vocab = list(tokenizer.vocab.keys())  # use and only use the token w2i in bert
        except:
            vocab = list(tokenizer.get_vocab().keys())  # use and only use the token w2i in bert
        self.vocab = self.vocab_cls(counter, specials=vocab, **kwargs)

    def pad(self, minibatch):  # 对一个二维（每行不定长进行padding）,需要batch padding和 len padding
        minibatch = list(minibatch)
        if not self.sequential:  # 已经序列化过的数值数据，不用padding
            return minibatch
        if self.fix_length is None:  # 没有指定长度，取batch中最长
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length
        padded, lengths = [], []
        for x in minibatch:
            #print("Before Padding is {}".format(x))
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len])
                )
            else:
                padded.append(
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    [self.pad_token] * max(0, max_len - len(x))
                )
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None, train=True):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x]for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)

        var = torch.tensor(arr, dtype=torch.long, device=device)
        if self.include_lengths:
            return var, lengths
        return var

class MultiTokenField(Field):
    '''
    Processing data like "[ ["A", "A", "A"], ["A", "A"], ["A", "A"], ["A"] ]"
    '''

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):  # never
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):  # never
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = [Pipeline(six.text_type.lower)(xx) for xx in x]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                for xx in x:
                    counter.update(xx)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):  # 对一个二维（每行不定长进行padding）,需要batch padding和 len padding
        minibatch = list(minibatch)
        if not self.sequential:  # 已经序列化过的数值数据，不用padding
            return minibatch
        if self.fix_length is None:  # 没有指定长度，取batch中最长
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2  # 否则固定长度还要去掉first和end_token（若有)
        padded, lengths = [], []
        for x in minibatch:
            #print("Before Padding is {}".format(x))
            if self.pad_first:
                padded.append(
                    [[self.pad_token]] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [[self.init_token]]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [[self.eos_token]]))
            else:
                padded.append(
                    ([] if self.init_token is None else [[self.init_token]]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [[self.eos_token]]) +
                    [[self.pad_token]] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
            #print("after padding is {}".format(padded[-1]))

        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr, device=None, train=True):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[[self.vocab.stoi[xx] for xx in x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)

        if self.include_lengths:
            return arr, lengths
        return arr

class MultiTokenTensorField(MultiTokenField):
    """similar as MultiTokenField But trans into tensor"""
    def pad(self, minibatch):  # 对一个二维（每行不定长进行padding）,需要batch padding和 len padding
        minibatch = list(minibatch)
        if not self.sequential:  # 已经序列化过的数值数据，不用padding
            return minibatch

        max_sen_len = max(len(x) for x in minibatch)
        max_unit_len = max(max([len(xx) for xx in x]) for x in minibatch)
        if self.fix_length is not None:  # 没有指定长度，取batch中最长
            if self.fix_length[0] is not None:
                max_sen_len = self.fix_length[0]
            if self.fix_length[1] is not None:
                max_unit_len = self.fix_length[1]

        padded, lengths = [], []
        pad_line = [self.pad_token] * max_unit_len
        #print("max_sen_len = {}, max_unit_len = {}, pad_line is{}".format(max_sen_len, max_unit_len, pad_line, type(pad_line)))
        for x in minibatch:
            #print("Before Padding is {}".format(x))
            lengths.append([])
            for idx in range(len(x)):
                lengths[-1] += [len(x[idx])]
                x[idx] = x[idx] + [self.pad_token] * max(0, max_unit_len - len(x[idx]))
            x = list(x[-max_sen_len:] if self.truncate_first else x[:max_sen_len])
            if self.pad_first:
                padded.append([pad_line] * max(0, max_sen_len - len(x)) + x)
                lengths[-1] = [0] * max(0, max_sen_len - len(x)) + lengths[-1]
            else:
                padded.append(x + [pad_line] * max(0, max_sen_len - len(x)))
                lengths[-1] += [0] * max(0, max_sen_len - len(x))
            #print("after padding is {}".format(padded[-1]))
            #print("length is {}".format(lengths))

        if self.include_lengths:
            return (padded, lengths) # length = batch * sen_len
        return padded  # 应该是batch* sen_len * unit_len

    def numericalize(self, arr, device=None, train=True):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        #for ex in arr: print(arr)
        if self.use_vocab:
            if self.sequential:
                arr = [[[self.vocab.stoi[xx] if xx in self.vocab.stoi.keys() else self.vocab.stoi[self.unk_token]
                         for xx in x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [[numericalization_func(xx) if isinstance(xx, six.string_types) else xx for xx in x
                        ] for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        #print(arr, len(arr), len(arr[0]))
        var = torch.tensor(arr, dtype=self.dtype, device=device)
        if self.include_lengths:
            return var, lengths
        return var