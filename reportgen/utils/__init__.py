# -*- coding: utf-8
'''
存在一些工具
'''
from .utils import iqr
from .metrics import WeightOfEvidence
from .metrics import entropy
from .metrics import entropyc
from .metrics import entropyd
from .discretization import Discretization

__all__=['iqr',
'WeightOfEvidence',
'entropy',
'entropyc',
'entropyd',
'Discretization']
