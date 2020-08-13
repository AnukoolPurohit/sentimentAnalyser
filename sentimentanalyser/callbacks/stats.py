from sentimentanalyser.callbacks.core import Callback
from sentimentanalyser.utils.data import listify
from fastprogress.fastprogress import format_time
from functools import partial
import time
import torch


class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train
    
    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets
    
    @property
    def avg_stats(self):
        return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    
    def accumulate(self, trainer):
        bs = trainer.xb.shape[0]
        self.tot_loss += trainer.loss * bs
        self.count += bs
        for i,metric in enumerate(self.metrics):
            self.tot_mets[i] += metric(trainer.preds, trainer.yb) * bs


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)
    
    def begin_fit(self):
        met_names = ['loss']
        for metric in self.train_stats.metrics:
            if isinstance(metric, partial):
                met_names.append(metric.func.__name__)
            else:
                met_names.append(metric.__name__)
        names = ['epoch'] + [f'train_{name}' for name in met_names] + [
            f'valid_{name}' for name in met_names] + ['time']
        self.logger(names)
        return
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        return
    
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.trainer)
    
    def after_epoch(self):
        stats = [str(self.epoch)]
        for s in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in s.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)