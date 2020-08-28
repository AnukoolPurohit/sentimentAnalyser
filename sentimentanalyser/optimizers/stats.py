import torch


class Stat:
    _defaults = {}

    def init_state(self, p):
        raise NotImplementedError

    def update(self, p, state, **kwargs):
        raise NotImplementedError


class AverageGrad(Stat):
    _defaults = dict(mom=0.9)

    def __init__(self, dampening: bool = False):
        self.dampening = dampening

    def init_state(self, p):
        return {'grad_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, mom, **kwargs):
        state['mom_damp'] = 1 - mom if self.dampening else 1.
        state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
        return state


class AverageSqrGrad(Stat):
    _defaults = dict(sqr_mom=0.99)

    def __init__(self, dampening: bool = True):
        self.dampening = dampening

    def init_state(self, p):
        return {'sqr_avg': torch.zeros_like(p.grad.data)}

    def update(self, p, state, sqr_mom, **kwargs):
        state['sqr_damp'] = 1 - sqr_mom if self.dampening else 1.
        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)
        return state


class StepCount(Stat):
    def init_state(self, p):
        return {'step': 0}

    def update(self, p, state, **kwargs):
        state['step'] += 1
        return state
