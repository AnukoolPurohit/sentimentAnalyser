import matplotlib.pyplot as plt
from .core import Callback


class Recorder(Callback):
    """
        Recorder class, records hyper-parameters and losses during Model fitting.
        This class is designed for torch.optim optimizers.
    """
    def begin_fit(self):
        self.train_losses = []
        self.valid_losses = []
        self.lrs = [[] for _ in self.opt.param_groups]
        return

    def after_batch(self):
        if not self.in_train:
            self.valid_losses.append(self.loss.detach().cpu())
            return

        self.train_losses.append(self.loss.detach().cpu())
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg['lr'])
        return

    def plot(self, skip_last=0, pgid=-1):
        losses = [loss.item() for loss in self.train_losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last

        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])
        return

    def plot_loss(self, skip_last=0):

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(self.train_losses[:len(self.train_losses) - skip_last],
                   'b', label='Training Loss')
        ax[0].legend(loc='best')

        ax[1].plot(self.valid_losses[:len(self.valid_losses) - skip_last],
                   'y', label='Validation Loss')
        ax[1].legend(loc='best')

        fig.suptitle('Training Summary')

    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])
        plt.xlabel('iterations')
        plt.ylabel('learning rate')
        plt.title('learning rate schedule')


class RecorderCustom(Callback):
    def begin_fit(self):
        self.train_losses = []
        self.valid_losses = []
        self.lrs = []
        return

    def after_batch(self):
        if not self.in_train:
            self.valid_losses.append(self.loss.detach().cpu())
            return

        self.train_losses.append(self.loss.detach().cpu())
        self.lrs.append(self.opt.hypers[-1]['lr'])
        return

    def plot(self, skip_last=0):
        losses = [loss.item() for loss in self.train_losses]
        n = len(losses) - skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])
        return

    def plot_loss(self, skip_last=0):

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(self.train_losses[:len(self.train_losses) - skip_last],
                   'b', label='Training Loss')
        ax[0].legend(loc='best')

        ax[1].plot(self.valid_losses[:len(self.valid_losses) - skip_last],
                   'y', label='Validation Loss')
        ax[1].legend(loc='best')

        fig.suptitle('Training Summary')

    def plot_lr(self):
        plt.plot(self.lrs)
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')
        plt.title('Learning rate schedule')
