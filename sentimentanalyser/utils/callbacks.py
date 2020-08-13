import re
import math
import torch
from functools import partial
from sentimentanalyser.utils.data import listify

def camel2snake(name):
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2',s1).lower()

def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos):
    return start + pos(end-start)

@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start)/2

@annealer
def sched_no(start, end, pos):
    return start

@annealer
def sched_exp(start, end, pos):
    return start * (end/start) ** pos


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >=0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = torch.nonzero((pos >= pcts), as_tuple=False).max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx+1] - pcts[idx])
        return scheds[idx](actual_pos)
    return _inner