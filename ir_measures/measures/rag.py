from ir_measures import measures
from .base import Measure, ParamInfo

class _EM(measures.Measure):

    __name__ = "EM"
    NAME = __name__
    PRETTY_NAME = "Exact Match"
    SHORT_DESC = "exact match of predicted answer and gold answers."
    SUPPORTED_PARAMS = {}

class _F1(measures.Measure):

    __name__ = "F1"
    NAME = __name__
    PRETTY_NAME = "f1 score"
    SHORT_DESC = "f1 score of predicted answer and ground-truth answer."
    SUPPORTED_PARAMS = {}

EM = _EM()
exact_match = EM
measures.register(EM, ["exact_match"])

F1 = _F1()
f1 = F1 
measures.register(F1, ["f1"])

