import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(init=False)
class PolarizationBasis:
    name: str
    type: str
    length: int


class JonesMatrix(ABC):

    def __init__(self):
        # private parent variables
        self._shape = None
        self._parameters = None
        self._matrix = None

        # public parent variable
        self.type = None
        self.dtype = None
        self.n_time = None
        self.n_channel_parameters = None
        self.n_channel_matrices = None
        self.n_parameters = None
        self.parameters = None
        self.matrix = None

        self.polarization_basis = PolarizationBasis()
        self.name = "JonesMatrix"

    @property
    @abstractmethod
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    @abstractmethod
    def shape(self, shape: tuple):
        self._shape = shape

    @property
    @abstractmethod
    def parameters(self) -> None:
        return self._parameters

    @property
    @parameters.setter
    def parameters(self, array: np.array) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        raise NotImplementedError

    @matrix.setter
    @abstractmethod
    def matrix(self, array: np.array) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def file_name(self) -> str:
        raise NotImplementedError
