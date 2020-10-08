from ..utils.data import listify
from .core import Callback


class ParamScheduler(Callback):
    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs
    
    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)
    
    def set_params(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg,sched_func in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = sched_func(self.n_epochs/self.epochs)
    
    def begin_batch(self):
        if self.in_train:
            self.set_params()


class ParamSchedulerCustom(Callback):
    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, listify(sched_funcs)
    
    def begin_batch(self):
        if not self.in_train:
            return
        fs = self.sched_funcs
        if len(fs) == 1:
            fs = fs*len(self.opt.param_groups)
        pos = self.n_epochs/self.epochs
        for f, h in zip(fs, self.opt.hypers):
            h[self.pname] = f(pos)
