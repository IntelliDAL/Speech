# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/053_optimizer.ipynb (unless otherwise specified).

__all__ = ['wrap_optimizer']

# Cell
from functools import partial
from fastai.optimizer import *

# Cell
def wrap_optimizer(opt, **kwargs): return partial(OptimWrapper, opt=opt, **kwargs)