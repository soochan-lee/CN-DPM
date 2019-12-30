from .ndpm_model import NdpmModel
from .singleton_model import SingletonModel
from .reservoir_model import ReservoirModel

MODEL = {
    'ndpm_model': NdpmModel,
    'singleton_model': SingletonModel,
    'reservoir_model': ReservoirModel,
}
