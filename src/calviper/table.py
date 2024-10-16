import numpy as np
import xarray as xr

import toolviper.utils.logger as logger

from abc import ABC
from abc import abstractmethod

from typing import Union

class BaseCalibrationTable(ABC):

    # Base calibration table abstract class
    @abstractmethod
    def generate(self, coords: dict)->Union[xr.Dataset, None]:
        pass

class CalibrationFactory(ABC):
    # Base factory class for table factory
    @abstractmethod
    def create_table(self, factory: Union[None, str]):
        pass

class GainTable(BaseCalibrationTable):

    # This is intended to be an implementation of a gain table simulator. It is
    # currently very rough and filled with random numbers. Generally based on the
    # original cal.py
    def generate(self, coords: dict)-> xr.Dataset:
        shape = tuple(value.shape[0] for value in coords.values())

        dims = {}
        for key, value in coords.items():
            dims[key] = value.shape[0]

        parameter = np.random.uniform(-np.pi, np.pi, shape)
        amplitude = np.random.normal(1.0, 0.1, shape)
        parameter = np.vectorize(complex)(
            np.cos(parameter),
            np.sin(parameter)
        )

        xds = xr.Dataset()

        xds["PARAMETER"] = xr.DataArray(amplitude * parameter, dims=dims)
        xds = xds.assign_coords(coords)

        return xds


    @staticmethod
    def empty_like(dataset: xr.Dataset)->xr.Dataset:
        dims = list(dataset.sizes)
        parameter = np.empty(tuple(dataset.sizes.values()))

        xds = xr.Dataset()

        xds["PARAMETER"] = xr.DataArray(parameter, dims=dims)
        xds.attrs["calibration_type"] = "gain"

        xds = xds.assign_coords(dataset.coords)

        return xds


class CalibrationTable(CalibrationFactory):

    def __init__(self):
        self.factory_list = {
            "gain": GainTable,
        }

    def create_table(self, factory: Union[None, str])->Union[BaseCalibrationTable, None]:
        try:
            return self.factory_list[factory]()

        except KeyError:
            logger.error(f"Factory method, {factory} not implemented.")
            return None