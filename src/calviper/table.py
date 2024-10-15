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

    # Specific implementation of gain table
    def generate(self, coords: dict)-> Union[xr.Dataset, None]:
        pass

    def empty_like(self, dataset: xr.Dataset)->Union[xr.Dataset, None]:
        pass

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