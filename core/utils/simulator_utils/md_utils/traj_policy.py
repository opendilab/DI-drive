import numpy as np

from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi, point_distance
from metadrive.utils import norm


class TrajPolicy(BasePolicy):

    def __init__(self, control_object, random_seed):
        super(TrajPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.last_heading = 0

    def convert_wp_to_world_coord(self, rbt_pos, rbt_heading, wp, visual=False):
        compose_visual = 0
        if visual:
            compose_visual += 0  # 4.51 / 2
        theta = np.arctan2(wp[1], wp[0] + compose_visual)
        rbt_heading = rbt_heading  # np.arctan2(rbt_heading[1], rbt_heading[0])
        theta = wrap_to_pi(rbt_heading) + wrap_to_pi(theta)
        norm_len = norm(wp[0] + compose_visual, wp[1])
        position = rbt_pos
        #position += 4.51 /2
        heading = np.sin(theta) * norm_len
        side = np.cos(theta) * norm_len
        return position[0] + side, position[1] + heading

    def convert_waypoint_list_coord(self, rbt_pos, rbt_heading, wp_list, visual=False):
        wp_w_list = []
        LENGTH = 4.51
        for wp in wp_list:
            # wp[0] = wp[0] + LENGTH / 2
            wp_w = self.convert_wp_to_world_coord(rbt_pos, rbt_heading, wp, visual)
            wp_w_list.append(wp_w)
        return wp_w_list

    def act(self, *args, **kwargs):
        if (self.control_object.arrive_destination and hasattr(self.control_object, 'macro_succ')):
            self.control_object.macro_succ = True
        if (self.control_object.crash_vehicle and hasattr(self.control_object, 'crash_vehicle')):
            self.control_object.macro_crash = True
        if (len(args) >= 2):
            macro_action = args[1]
        frame = args[1]
        wp_list = args[2]
        ego_vehicle = self.control_object
        if frame == 0:
            self.base_pos = ego_vehicle.position
            self.base_heading = ego_vehicle.heading_theta
            self.control_object.v_wps = self.convert_waypoint_list_coord(
                self.base_pos, self.base_heading, wp_list, True
            )
            self.control_object.penultimate_state = self.control_object.traj_wp_list[
                -2]  # if len(wp_list)>2 else self.control_object.traj_wp_list[-1]
            new_state = {}
            new_state['position'] = ego_vehicle.position
            new_state['yaw'] = ego_vehicle.heading_theta
            new_state['speed'] = ego_vehicle.last_spd
            self.control_object.traj_wp_list = []
            self.control_object.traj_wp_list.append(new_state)
        self.control_object.v_indx = frame
        wp_list = self.convert_waypoint_list_coord(self.base_pos, self.base_heading, wp_list)
        current_pos = np.array(wp_list[frame])
        target_pos = np.array(wp_list[frame + 1])
        diff = target_pos - current_pos
        norm = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

        if abs(norm) < 0.001:
            heading_theta_at = self.last_heading
        else:
            direction = diff / norm
            heading_theta_at = np.arctan2(direction[1], direction[0])
        self.last_heading = heading_theta_at
        steering = 0  # self.steering_conrol_traj(lateral, heading_theta_at)
        throtle_brake = 0  # self.speed_control(target_vel)
        # print('target_vel: {}'.format(target_vel))
        # print('target_heading_theta: {}'.format(heading_theta_at))
        ttarget_pos = self.base_pos + target_pos
        hheading_theata_at = heading_theta_at + self.base_heading
        # print('target frame: {} with position ({}, {}) and orientation {}'.format(
        #     frame, target_pos[0], target_pos[1], heading_theta_at))

        # onestep state update, for trex
        ego_vehicle.prev_state['position'] = ego_vehicle.curr_state['position']
        ego_vehicle.prev_state['yaw'] = ego_vehicle.curr_state['yaw']
        ego_vehicle.prev_state['speed'] = ego_vehicle.curr_state['speed']

        ego_vehicle.set_position(target_pos)
        ego_vehicle.set_heading_theta(heading_theta_at)
        ego_vehicle.last_spd = norm / ego_vehicle.physics_world_step_size
        new_state = {}
        new_state['position'] = target_pos
        new_state['yaw'] = heading_theta_at
        new_state['speed'] = ego_vehicle.last_spd
        self.control_object.traj_wp_list.append(new_state)

        # onestep state update, for trex
        ego_vehicle.curr_state['position'] = target_pos
        ego_vehicle.curr_state['yaw'] = heading_theta_at
        ego_vehicle.curr_state['speed'] = ego_vehicle.last_spd

        #print(ego_vehicle.physics_world_step_size)
        #ego_vehicle.last_spd = norm / 0.03 * 3.6
        #ego_vehicle.set_velocity(heading_theta_at, norm / 0.03 *3.6)

        #throtle_brake = throtle_brake if self.stop_label is False else -1
        return [steering, throtle_brake]
