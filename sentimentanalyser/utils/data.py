import re
import torch
import numpy as np
from typing import Iterable
from pathlib import Path
from tqdm.autonotebook import tqdm
from torch import LongTensor, tensor
from concurrent.futures import ProcessPoolExecutor
from sentimentanalyser.preprocessing.tokens import TOKENS


def listify(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    if isinstance(x, Iterable):
        return list(x)
    return [x]


def compose(x, funcs, order_attr="_order", **kwargs):
    key = lambda o: getattr(o, order_attr, 0)
    for func in sorted(listify(funcs), key=key):
        x = func(x, **kwargs)
    return x


def read_text_file(file):
    with open(file, 'r', encoding='utf8') as f:
        return f.read()


def filter_files(files, include=[], exclude=[], attr=None):
    if attr is None:
        fn = lambda x: str(x)
    else:
        fn = lambda x: getattr(x, attr)
    for incl in include:
        files = [file for file in files if incl in fn(file)]
    for excl in exclude:
        files = [file for file in files if excl not in fn(file)]
    return files


def ls(self, recurse=False, include=[], exclude=[], **kwargs):
    if recurse:
        files = list(self.glob("**/*"))
    else:
        files = list(self.iterdir())
    return filter_files(files, include, exclude, **kwargs)


Path.ls = ls


def random_splitter(file, p=[0.8, 0.1, 0.1]):
    if isinstance(p, float):
        p = sorted([p, 1-p], reverse=True)
    assert sum(p) == 1.
    return np.random.choice(list(range(len(p))), p=sorted(p, reverse=True))


def grandparent_splitter(file, valid_name=None, test_name=None):
    if file.parent.parent.name is valid_name:
        return 1
    elif file.parent.parent.name is test_name:
        return 2
    else:
        return 0


def parallel(func, arr, max_workers=4):
    if max_workers < 2:
        results = list(map(func, tqdm(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(tqdm(ex.map(func, arr), total=len(arr), leave=False))
    if any([o is not None for o in results]):
        return results


def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i, s in enumerate(samples):
        if pad_first:
            res[i, -len(s[0]):] = LongTensor(s[0])
        else:
            res[i, :len(s[0])] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])


def parent_labeler(path):
    return path.parent.name


def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0


def read_wiki(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i, line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            current_article = current_article.replace('<unk>', TOKENS.UNK)
            articles.append(current_article)
            current_article = ''
    current_article = current_article.replace('<unk>', TOKENS.UNK)
    articles.append(current_article)
    return articles
