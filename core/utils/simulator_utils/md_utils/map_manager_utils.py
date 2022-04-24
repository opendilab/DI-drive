from metadrive.manager.map_manager import MapManager
from metadrive.component.map.pg_map import PGMap

MIN_LANE_NUM = 4
MAX_LANE_NUM = 6


class MacroMapManager(MapManager):

    def add_random_to_map(self, map_config):
        if self.engine.global_config["random_lane_width"]:
            map_config[PGMap.LANE_WIDTH
                       ] = self.np_random.rand() * (PGMap.MAX_LANE_WIDTH - PGMap.MIN_LANE_WIDTH) + PGMap.MIN_LANE_WIDTH
        if self.engine.global_config["random_lane_num"]:
            map_config[PGMap.LANE_NUM] = self.np_random.randint(MIN_LANE_NUM, MAX_LANE_NUM + 1)
        return map_config
