
import torch
from sentimentanalyser.callbacks.training import TrainEvalCallback
from sentimentanalyser.utils.exceptions import CancelTrainException
from sentimentanalyser.utils.exceptions import CancelEpochException
from sentimentanalyser.utils.exceptions import CancelBatchException
from sentimentanalyser.utils.data import listify

class Trainer():
    def __init__(self, data, model, loss_func, opt, cbs=None, cb_funcs=None):
        self.data, self.model    = data, model
        self.loss_func, self.opt = loss_func, opt
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))
        return
    
    def add_cb(self, cb):
        cb.set_trainer(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
        return
    
    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)
        return
    
    def remove_cbs(self, cbs):
        for cb in listify(cbs):
            self.cbs.remove(cb)
        return
    
    def one_batch(self, itr, xb, yb):
        try:
            self.iter = itr
            self.xb, self.yb = xb, yb
            self('begin_batch')
            self.preds  = self.model(self.xb)
            self('after_pred')
            self.loss   = self.loss_func(self.preds, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally:
            self('after_batch')
        return
    
    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for itr, (xb, yb) in enumerate(self.dl):
                self.one_batch(itr, xb, yb)
        except CancelEpochException:
            self('after_cancel_epoch')
        return 
    
    def fit(self, epochs=3):
        self.epochs, self.loss = epochs, 0.
        try:
            for cb in self.cbs:
                cb.set_trainer(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                self.dl    = self.data.train_dl
                # Training Loop
                if not self('begin_epoch'):
                    self.all_batches()

                # Validation Loop
                self.dl    = self.data.valid_dl
                with torch.no_grad():
                    if not self('begin_validate'):
                        self.all_batches()
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')
    
    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
        'begin_epoch', 'begin_epoch', 'begin_validate', 'after_epoch',
        'after_cancel_train', 'after_fit'}
    
    def __call__(self, cb_name):
        res = True
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res
    
    @property
    def pred(self):
        return self.preds