from sentimentanalyser.callbacks.core import Callback
from sentimentanalyser.utils.exceptions import CancelTrainException
from torch import nn


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.trainer.n_epochs = 0.
        self.trainer.n_iter   = 0
    
    def after_batch(self):
        if not self.in_train: return
        self.trainer.n_epochs += 1./self.iters
        self.trainer.n_iter           += 1
        
    def begin_epoch(self):
        self.trainer.n_epochs = self.epoch 
        self.model.train()
        self.trainer.in_train = True
        return
    def begin_validate(self):
        self.model.eval()
        self.trainer.in_train = False
        return


class CudaCallback(Callback):
    def begin_fit(self):
        self.model.cuda()
    
    def begin_batch(self):
        self.trainer.xb = self.xb.cuda()
        self.trainer.yb = self.yb.cuda()


class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9
    
    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr  = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return
    
    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class GradientClipping(Callback):
    def __init__(self, clip=None):
        self.clip = clip
    
    def after_backward(self):
        if self.clip:
            nn.utils.clip_grad_norm_(self.trainer.model.parameters(), self.clip)