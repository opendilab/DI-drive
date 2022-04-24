from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, safe_clip_for_small_array
from typing import Union, Dict, AnyStr, Tuple


class MacroBaseVehicle(BaseVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroBaseVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
