from sentimentanalyser.utils.data import listify, compose, read_text_file, Path, random_splitter
from sentimentanalyser.utils.data import pad_collate
from sentimentanalyser.data.samplers import SortishSampler, SortSampler
from torch.utils.data import DataLoader
from torch import tensor
from typing import Any
import torch

class ListContainer():
    def __init__(self, items):
        self.items = listify(items)
    
    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)
                return [v for v,m in zip(self.items, idx) if m]
            return [self.items[i] for i in idx]
    
    def __len__(self):
        return len(self.items)
    
    def __iter__(self):
        return iter(self.items)
    
    def __delitem__(self, idx):
        del(self.items[idx])
    
    def __setitem__(self, idx, value):
        self.items[idx] = value
    
    @staticmethod
    def display_lists(lists):
        res = ""
        for lst in lists:
            res += f"list ({len(lst)} items) {lst[:5].__repr__()[:-1]}…]\t"
        return res

    def __repr__(self):
        if self.items == []:
            ret = f"{self.__class__.__name__} ({len(self)} items)\n{self.items}"
        elif isinstance(self.items[0], list):
            disp_lst = self.display_lists(self.items[:10])
            ret = f"{self.__class__.__name__} ({len(self)} items)\n{disp_lst}"
        else:
            ret = f"{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            return f"{ret[:-1]}……]"
        else:
            return ret


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
        mask = [func(item) for item in self.items]
        train = [item for item,m in zip(self.items, mask) if m == 0]
        valid = [item for item,m in zip(self.items, mask) if m == 1]
        test  = [item for item,m in zip(self.items, mask) if m == 2]
        return SplitData(self.new(train), self.new(valid), self.new(test)) 


class TextList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, folders=[], **kwargs):
        files = []
        if extensions is None:
            extensions=['.txt']
        extensions = listify(extensions)
        for folder in folders:
            files += (path/folder).ls(recurse=True, include=extensions, attr='suffix')
        return cls(files, path, **kwargs)
    
    def get(self, i):
        if isinstance(i, Path):
            return read_text_file(i)
        return i


class SplitData():
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
    
    def __setstate__(self, data:Any):
        self.__dict__.update(data)
    
    def label_by_func(self, func, proc_x=None, proc_y=None):
        train = LabeledData.label_by_func(self.train, func, proc_x=proc_x, proc_y=proc_y)
        valid = LabeledData.label_by_func(self.valid, func, proc_x=proc_x, proc_y=proc_y)
        if self.test is not None:
            test  = LabeledData.label_by_func(self.test, func, proc_x=proc_x, proc_y=proc_y)
        else:
            test = None
        cls = self.__class__
        return cls(train, valid, test)
    
    def lm_databunchify(self, bs, bptt, **kwargs):
        train_dl = DataLoader(LM_Dataset(self.train, bs, bptt, shuffle=True),
                                batch_size=bs,  **kwargs)
        valid_dl = DataLoader(LM_Dataset(self.valid, bs, bptt, shuffle=False),
                                batch_size=bs, **kwargs)
        if self.test is None:
            test_dl = None
        elif len(self.test) >0:
            test_dl = DataLoader(LM_Dataset(self.test, bs, bptt, shuffle=False, batch_size=bs, **kwargs))
        else:
            test_dl = None
        return DataBunch(train_dl, valid_dl, test_dl)
    
    def clas_databunchify(sd, bs, **kwargs):
        train_sampler = SortishSampler(sd.train.x, key=lambda t: len(sd.train.x[t]), bs=bs)
        valid_sampler = SortSampler(sd.valid.x, key=lambda t: len(sd.valid.x[t]))
        train_dl = DataLoader(sd.train, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate, **kwargs)
        valid_dl = DataLoader(sd.valid, batch_size=bs*2, sampler=valid_sampler, collate_fn=pad_collate, **kwargs)
        return DataBunch(train_dl, valid_dl)


class LabeledData():
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
        return (self.x[idx], self.y[idx])

    @staticmethod
    def _label_by_func(func, il, cls=ItemList):
        return cls([func(item) for item in il.items], il.path)

    @classmethod
    def label_by_func(cls, il, func, proc_x=None, proc_y=None):
        return cls(il, cls._label_by_func(func, il), proc_x=proc_x, proc_y=proc_y)


class LM_Dataset():
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
        self.batched_data = stream[:self.n_batches*self.bs].view(self.bs, self.n_batches)
    
    def __len__(self):
        return ((self.n_batches - 1) // self.bptt) * self.bs
    
    def __getitem__(self, idx):
        source = self.batched_data[idx%self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return source[seq_idx: seq_idx+self.bptt], source[seq_idx+1: seq_idx+self.bptt+1]


class DataBunch():
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