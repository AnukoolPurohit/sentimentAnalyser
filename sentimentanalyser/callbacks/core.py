import re
from .utils import camel2snake


class Callback:
    """
        Core Callback class that will be inherited by all callbacks.
    """
    _order = 0

    def set_trainer(self, trainer):
        """
            Gives callback access to the trainer object and it's attributes.
        """
        self.trainer = trainer
        return
    
    def set_runner(self, run):
        """
            Same functionality as set_trainer except adds compatibility with fastai callbacks for testing.
        """
        self.trainer = run
        return
        
    def __getattr__(self, k):
        """
            If you can't find the specified attribute in the Callback, look for it in Trainer.
        """
        return getattr(self.trainer, k)
    
    def __call__(self, cbname):
        """
            Call attribute functions of a callback by calling the callback and passing the function name as string.
        """
        cb_func = getattr(self, cbname, None)
        if cb_func and cb_func():
            return True
        return False
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
