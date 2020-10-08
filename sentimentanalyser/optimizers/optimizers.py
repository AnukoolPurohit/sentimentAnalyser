from functools import partial
from ..utils.data import listify
from .core import StatefulOptimizer
from .steppers import adam_step, weight_decay
from .stats import AverageGrad, AverageSqrGrad, StepCount


def adam_opt(xtra_step=None, **kwargs):
    return partial(StatefulOptimizer, steppers=[adam_step, weight_decay]+listify(xtra_step),
                   stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()], **kwargs)
