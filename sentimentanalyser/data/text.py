from ..utils.data import listify, compose, read_text_file, Path, random_splitter
from ..utils.data import pad_collate
from .samplers import SortishSampler, SortSampler
from .core import ListContainer
from torch.utils.data import DataLoader
from torch import tensor
from typing import Any
import torch


class ItemList(ListContainer):
    def __init__(self, items, path, tfms=None):
        super().__init__(items)
        self.path = path
        self.tfms = listify(tfms)

    def new(self, items, cls=None):
        if cls is None:
            cls = self.__class__
        return cls(items, self.path, self.tfms)

    def get(self, i):
        return i

    def _get(self, i):
        return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if isinstance(item, list):
            return [self._get(i) for i in item]
        return self._get(item)

    def __repr__(self):
        ret = f"{super().__repr__()}\nPath: \'{self.path}\'"
        return ret

    def split_by_func(self, func=random_splitter):
        #       Train  Valid  Test
        sets = {0: [], 1: [], 2: []}
        for item in self.items:
            sets[func(item)].append(item)
        return SplitData(self.new(sets[0]), self.new(sets[1]), self.new(sets[2]))


class TextList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, folders=None, **kwargs):
        files = []
        if not folders:
            folders = []
        if extensions is None:
            extensions = ['.txt']
        extensions = listify(extensions)
        for folder in folders:
            files += (path / folder).ls(recurse=True, include=extensions, attr='suffix')
        return cls(files, path, **kwargs)

    def get(self, i):
        if isinstance(i, Path):
            return read_text_file(i)
        return i


class SplitData:
    def __init__(self, train, valid, test=None):
        self.train, self.valid, self.test = train, valid, test

    @classmethod
    def from_function(cls, func, il):
        return cls(*func(il))

    def __getattr__(self, k):
        return getattr(self.train, k)

    def __repr__(self):
        msg_train = self.train.__repr__()
        msg_valid = self.valid.__repr__()
        if self.test is not None:
            msg_test = self.test.__repr__()
        else:
            msg_test = 'EMPTY'
        msg = f"Train:\n{msg_train}\n\nValid:\n{msg_valid}\n\nTest:\n{msg_test}"
        return msg

    def __setstate__(self, data: Any):
        self.__dict__.update(data)

    def label_by_func(self, func, proc_x=None, proc_y=None):
        train = LabeledDataset.label_by_func(self.train, func, proc_x=proc_x, proc_y=proc_y)
        valid = LabeledDataset.label_by_func(self.valid, func, proc_x=proc_x, proc_y=proc_y)
        if self.test is not None:
            test = LabeledDataset.label_by_func(self.test, func, proc_x=proc_x, proc_y=proc_y)
        else:
            test = None
        cls = self.__class__
        return cls(train, valid, test)

    def lm_databunchify(self, bs, bptt, **kwargs):
        train_dl = DataLoader(LangModelDataset(self.train, bs, bptt, shuffle=True),
                              batch_size=bs, **kwargs)
        valid_dl = DataLoader(LangModelDataset(self.valid, bs, bptt, shuffle=False),
                              batch_size=bs, **kwargs)
        if self.test is None:
            test_dl = None
        elif len(self.test) > 0:
            test_dl = DataLoader(LangModelDataset(self.test, bs, bptt, shuffle=False, batch_size=bs, **kwargs))
        else:
            test_dl = None
        return DataBunch(train_dl, valid_dl, test_dl)

    def clas_databunchify(self, bs, **kwargs):
        train_sampler = SortishSampler(self.train.x, key=lambda t: len(self.train.x[t]), bs=bs)
        valid_sampler = SortSampler(self.valid.x, key=lambda t: len(self.valid.x[t]))
        train_dl = DataLoader(self.train, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate, **kwargs)
        valid_dl = DataLoader(self.valid, batch_size=bs * 2, sampler=valid_sampler, collate_fn=pad_collate, **kwargs)
        return DataBunch(train_dl, valid_dl)


class LabeledDataset:
    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x = self.process(x, proc_x)
        self.y = self.process(y, proc_y)
        self.proc_x = proc_x
        self.proc_y = proc_y

    def process(self, il, process):
        return il.new(compose(il, process))

    def __repr__(self):
        return f"{self.__class__.__name__}\nX:\n{self.x}\nY:\n{self.y}"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def _label_by_func(func, il, cls=ItemList):
        return cls([func(item) for item in il.items], il.path)

    @classmethod
    def label_by_func(cls, il, func, proc_x=None, proc_y=None):
        return cls(il, cls._label_by_func(func, il), proc_x=proc_x, proc_y=proc_y)


class LangModelDataset:
    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data, self.bs, self.bptt, self.shuffle = data, bs, bptt, shuffle
        total_len = sum([len(t) for t in self.data.x])
        self.n_batches = total_len // bs
        self.batchify()

    def batchify(self):
        texts = self.data.x
        if self.shuffle:
            texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([tensor(t) for t in texts])
        self.batched_data = stream[:self.n_batches * self.bs].view(self.bs, self.n_batches)

    def __len__(self):
        return ((self.n_batches - 1) // self.bptt) * self.bs

    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return source[seq_idx: seq_idx + self.bptt], source[seq_idx + 1: seq_idx + self.bptt + 1]


class DataBunch:
    def __init__(self, train_dl, valid_dl, test_dl=None):
        self.train_dl, self.valid_dl, self.test_dl = train_dl, valid_dl, test_dl

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset

    @property
    def test_ds(self):
        if self.test_dl is None:
            return None
        else:
            return self.test_dl.dataset
