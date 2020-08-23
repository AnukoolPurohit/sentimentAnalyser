import re
from sentimentanalyser.utils.callbacks import camel2snake


class Callback():
    _order = 0
    def set_trainer(self, trainer):
        self.trainer = trainer
        return
    
    def set_runner(self, run):
        self.trainer = run
        return
        
    def __getattr__(self, k):
        return getattr(self.trainer, k)
    
    def __call__(self, cbname):
        cb_func = getattr(self, cbname, None)
        if cb_func and cb_func():
            return True
        return False
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')