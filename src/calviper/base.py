import numpy as np

from toolviper.utils import logger
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Union

T = TypeVar('T', bound='Parent')


@dataclass(init=False)
class PolarizationBasis:
    name: str
    type: str
    length: int


class JonesMatrix(ABC):

    def __init__(self):
        # private parent variables
        self._parameters: Union[np.array, None] = None
        self._matrix: Union[np.array, None] = None

        # public parent variable
        self.type: Union[str, None] = None
        self.dtype: Union[type, None] = None
        self.n_times: Union[int, None] = None
        self.n_antennas: Union[int, None] = None
        self.n_channels: Union[int, None] = None
        self.n_polarizations: Union[int, None] = None
        self.n_channel_matrices: Union[int, None] = None
        self.n_parameters: Union[int, None] = None
        self.caltable_name: Union[str, None] = None
        self.channel_dependent_parameters: bool = False

        self.polarization_basis: PolarizationBasis = PolarizationBasis()
        self.name: str = "BaseJonesMatrix"

    # Inherited member properties
    @property
    @abstractmethod
    def shape(self) -> tuple:
        return self.n_times, self.n_antennas, self.n_channels, self.n_polarizations, self.n_parameters

    @shape.setter
    @abstractmethod
    def shape(self, shape: tuple):
        # Reset parameters and matrices
        self._parameters = np.empty([])
        self._matrix = np.empty([])

        self.n_times, self.n_antennas, self.n_channels, self.n_polarizations, self.n_parameters = shape

    @property
    @abstractmethod
    def parameters(self) -> None:
        return self._parameters

    @parameters.setter
    @abstractmethod
    def parameters(self, array: np.array) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    @abstractmethod
    def matrix(self, array: np.array) -> np.ndarray:
        raise NotImplementedError

    # Inherited method properties
    @classmethod
    def from_parameters(cls: Type[T], parameters: dict) -> T:
        import inspect

        obj = cls()
        updated_params = {}

        # This is a bit of a complicated way to do this BUT it should allow for a generic form of
        # from_parameters() for all child classes. I think ...
        for key, value in parameters.items():
            if key in inspect.getmembers(cls.__bases__[0], predicate=inspect.isfunction):
                updated_params[f"_{key}"] = value

            elif key in inspect.getmembers(cls, predicate=inspect.isfunction):
                updated_params[f"_{key}"] = value

            else:
                if key in cls().__dict__.keys():
                    updated_params[key] = value

                elif key in cls.__bases__[0]().__dict__.keys():
                    updated_params[key] = value

                else:
                    pass

        vars(obj).update(updated_params)

        return obj

    def initialize_parameters(self, dtype: np.dtype, shape: tuple = None):
        # Set data type
        self.type = dtype

        # Update shape is needed
        if shape is not None:
            self.shape = shape

        # Initialize the parameters to default
        self.parameters = np.ones(shape, dtype=dtype)

        # Reset Jones
        self.matrix = np.empty([])

    def initialize_jones(self, shape: tuple = None):
        if shape is not None:
            self.shape = shape

        self.matrix = np.identity(2, dtype=np.complex64)
        self.matrix = np.tile(self.matrix, [self.n_times, self.n_antennas, self.n_channel_matrices, 1, 1])

    def invert(self) -> Union[np.array, None]:
        if np.any(np.abs(np.linalg.det(self.matrix)) == 0.):
            logger.error(f"Jones matrix is singular: {np.linalg.det(self.matrix)}")
            return None

        return np.linalg.inv(self.matrix)

    def accumulate(self, other: Type[T]) -> T:
        # I think this could just be an overload of __mul__()
        return np.matmul(other.matrix, self.matrix, out=self.matrix)
