from fastprogress import master_bar, progress_bar
from functools import partial
from .core import Callback


class ProgressCallback(Callback):
    """
        Callback to implement progress bar during Model Training.
    """
    _order = -1

    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.trainer.logger = partial(self.mbar.write, table=True)
        return
    
    def after_fit(self): 
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.iter)

    def begin_epoch(self):
        self.set_pb()

    def begin_validate(self):
        self.set_pb()
    
    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)
        return
