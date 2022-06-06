import numpy as np
from panda3d.core import TransparencyAttrib, LineSegs, NodePath

from metadrive.component.road_network import Road
from metadrive.component.vehicle_navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.utils import norm
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.math_utils import wrap_to_pi
from metadrive.constants import RENDER_MODE_ONSCREEN, CamMask


class TrajNodeNavigation(NodeNetworkNavigation):

    def __init__(
        self,
        engine,
        show_navi_mark: bool = False,
        random_navi_mark_color=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        seq_traj_len=30,
        show_seq_traj=False,
    ):
        super(NodeNetworkNavigation,
              self).__init__(engine, show_navi_mark, random_navi_mark_color, show_dest_mark, show_line_to_dest)
        self._show_traj = show_seq_traj
        self.seq_traj_len = seq_traj_len
        if self._show_traj:
            self._init_trajs()

        #self.drawd = False

        self.LINE_TO_DEST_HEIGHT += 1
        self.activate_car_pos_marker = False

    def _init_trajs(self):
        for i in range(self.seq_traj_len):
            init_line = LineSegs()
            init_line.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7)
            self.__dict__['traj_{}'.format(i)] = NodePath(init_line.create())
            self.__dict__['traj_{}'.format(i)].reparentTo(self.origin)
        init_line = LineSegs()
        init_line.setColor(0.5, 0.5, 0.5, 0.7)
        self.current_pos_marker = NodePath(init_line.create())
        self.current_pos_marker.reparentTo(self.origin)

    def _draw_trajectories(self, wp_list):
        for i in range(self.seq_traj_len):
            lines = LineSegs()
            lines.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7)
            #lines.moveTo(panda_position(wp_list[i][0], self.LINE_TO_DEST_HEIGHT+4))
            lines.moveTo(panda_position((wp_list[i][0], wp_list[i][1]), self.LINE_TO_DEST_HEIGHT))
            lines.drawTo(panda_position((wp_list[i + 1][0], wp_list[i + 1][1]), self.LINE_TO_DEST_HEIGHT))
            lines.setThickness(2)
            self.__dict__['traj_{}'.format(i)].removeNode()
            self.__dict__['traj_{}'.format(i)] = NodePath(lines.create(False))
            self.__dict__['traj_{}'.format(i)].hide(CamMask.Shadow | CamMask.RgbCam)
            self.__dict__['traj_{}'.format(i)].reparentTo(self.origin)

    def convert_wp_to_world_coord(self, index, rbt_pos, rbt_heading, wp):
        theta = np.arctan2(wp[1], wp[0])
        rbt_heading = np.arctan2(rbt_heading[1], rbt_heading[0])
        theta = wrap_to_pi(rbt_heading) + wrap_to_pi(theta)
        norm_len = norm(wp[0], wp[1])
        position = rbt_pos
        heading = np.sin(theta) * norm_len
        side = np.cos(theta) * norm_len
        return position[0] + side, position[1] + heading

    def convert_waypoint_list_coord(self, rbt_pos, rbt_heading, wp_list):
        wp_w_list = []
        for wp in wp_list:
            wp_w = self.convert_wp_to_world_coord(0, rbt_pos, rbt_heading, wp)
            wp_w_list.append(wp_w)
        return wp_w_list

    # def draw_path(self, rbt_pos, rbt_heading):
    #     wp_list = self.get_waypoint_list()
    #     wp_list = self.convert_waypoint_list_coord(rbt_pos, rbt_heading, wp_list)
    #     self._draw_trajectories(wp_list)

    def draw_car_path(self, wp_list):
        for i in range(self.seq_traj_len):
            lines = LineSegs()
            lines.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7)
            #lines.moveTo(panda_position(wp_list[i][0], self.LINE_TO_DEST_HEIGHT+4))
            lines.moveTo(panda_position((wp_list[i][0], wp_list[i][1]), self.LINE_TO_DEST_HEIGHT))
            lines.drawTo(panda_position((wp_list[i + 1][0], wp_list[i + 1][1]), self.LINE_TO_DEST_HEIGHT))
            lines.setThickness(2)
            self.__dict__['traj_{}'.format(i)].removeNode()
            self.__dict__['traj_{}'.format(i)] = NodePath(lines.create(False))
            self.__dict__['traj_{}'.format(i)].hide(CamMask.Shadow | CamMask.RgbCam)
            self.__dict__['traj_{}'.format(i)].reparentTo(self.origin)

    def show_car_pos(self, wp_list, current_time_step):
        #print(current_time_step)
        cx = wp_list[current_time_step][0]
        cy = wp_list[current_time_step][1]
        ncx = wp_list[current_time_step + 1][0]
        ncy = wp_list[current_time_step + 1][1]
        theta = np.arctan2(ncy - cy, ncx - cx)
        theta = 0.0
        lines = LineSegs()
        lines.setColor(0.9, 0.9, 0.9, 1.0)
        #lines.moveTo(panda_position(wp_list[i][0], self.LINE_TO_DEST_HEIGHT+4))
        lines.moveTo(panda_position((cx - 0.1 * np.sin(theta), cy + 0.1 * np.cos(theta)), self.LINE_TO_DEST_HEIGHT))
        lines.drawTo(panda_position((cx + 0.1 * np.sin(theta), cy - 0.1 * np.cos(theta)), self.LINE_TO_DEST_HEIGHT))
        lines.setThickness(10)
        self.current_pos_marker.removeNode()
        self.current_pos_marker = NodePath(lines.create(False))
        self.current_pos_marker.hide(CamMask.Shadow | CamMask.RgbCam)
        self.current_pos_marker.reparentTo(self.origin)
        for i in range(self.seq_traj_len):
            lines = LineSegs()
            if current_time_step > i:
                lines.setColor(1.0, 0.4, 0.0, 0.7)
            else:
                lines.setColor(0.0, 0.7, 1.0, 0.7)
            #lines.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7)
            #lines.moveTo(panda_position(wp_list[i][0], self.LINE_TO_DEST_HEIGHT+4))
            lines.moveTo(panda_position((wp_list[i][0], wp_list[i][1]), self.LINE_TO_DEST_HEIGHT))
            lines.drawTo(panda_position((wp_list[i + 1][0], wp_list[i + 1][1]), self.LINE_TO_DEST_HEIGHT))
            lines.setThickness(2)
            self.__dict__['traj_{}'.format(i)].removeNode()
            self.__dict__['traj_{}'.format(i)] = NodePath(lines.create(False))
            self.__dict__['traj_{}'.format(i)].hide(CamMask.Shadow | CamMask.RgbCam)
            self.__dict__['traj_{}'.format(i)].reparentTo(self.origin)

    def get_waypoint_list(self):
        x = np.arange(0, 50, 0.1)
        LENGTH = 4.51
        y = 1 * np.cos(x) - 1
        x = x + LENGTH / 2
        lst = []
        for i in range(x.shape[0]):
            lst.append([x[i], y[i]])
        return lst

    def update_localization(self, ego_vehicle):
        position = ego_vehicle.position
        lane, lane_index = self._update_current_lane(ego_vehicle)
        long, _ = lane.local_coordinates(position)
        need_update = self._update_target_checkpoints(lane_index, long)
        assert len(self.checkpoints) >= 2

        # target_road_1 is the road segment the vehicle is driving on.
        if need_update:
            target_road_1_start = self.checkpoints[self._target_checkpoints_index[0]]
            target_road_1_end = self.checkpoints[self._target_checkpoints_index[0] + 1]
            target_lanes_1 = self.map.road_network.graph[target_road_1_start][target_road_1_end]
            self.current_ref_lanes = target_lanes_1
            self.current_road = Road(target_road_1_start, target_road_1_end)

            # target_road_2 is next road segment the vehicle should drive on.
            target_road_2_start = self.checkpoints[self._target_checkpoints_index[1]]
            target_road_2_end = self.checkpoints[self._target_checkpoints_index[1] + 1]
            target_lanes_2 = self.map.road_network.graph[target_road_2_start][target_road_2_end]

            if target_road_1_start == target_road_2_start:
                # When we are in the final road segment that there is no further road to drive on
                self.next_road = None
                self.next_ref_lanes = None
            else:
                self.next_road = Road(target_road_2_start, target_road_2_end)
                self.next_ref_lanes = target_lanes_2

        self._navi_info.fill(0.0)
        half = self.navigation_info_dim // 2
        self._navi_info[:half], lanes_heading1, checkpoint = self._get_info_for_checkpoint(
            lanes_id=0, ref_lane=self.current_ref_lanes[0], ego_vehicle=ego_vehicle
        )

        self._navi_info[half:], lanes_heading2, _ = self._get_info_for_checkpoint(
            lanes_id=1,
            ref_lane=self.next_ref_lanes[0] if self.next_ref_lanes is not None else self.current_ref_lanes[0],
            ego_vehicle=ego_vehicle
        )

        if hasattr(ego_vehicle, 'v_indx') and self._show_traj:
            #print(ego_vehicle.v_indx)
            if ego_vehicle.v_indx == 0:
                #self.draw_car_path(ego_vehicle.v_wps)
                self.activate_car_pos_marker = True
            if self.activate_car_pos_marker:
                self.show_car_pos(ego_vehicle.v_wps, ego_vehicle.v_indx)

            # if ego_vehicle.v_indx == 4:
            #     print('zt')

        # if ego_vehicle.v_indx == 0:
        #     self.draw_car_path(ego_vehicle.v_wps)

        # if self.drawd is False:
        #     self.draw_path(ego_vehicle.position, ego_vehicle.heading)
        #     self.drawd = True

        if self._show_navi_info:
            # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = checkpoint
            self._goal_node_path.setPos(pos_of_goal[0], -pos_of_goal[1], 1.8)
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)
            self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
            dest_pos = self._dest_node_path.getPos()
            #self._draw_line_to_dest(start_position=ego_vehicle.position, end_position=(dest_pos[0], -dest_pos[1]))
