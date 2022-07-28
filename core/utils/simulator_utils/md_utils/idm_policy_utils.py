from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
import logging


class MacroIDMPolicy(IDMPolicy):

    def __init__(self, control_object, random_seed, normal_speed=18, safe_lane_change_dist=15):
        super(MacroIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 20
        self.NORMAL_SPEED_CONST = normal_speed
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 300
        self.SAFE_LANE_CHANGE_DISTANCE = safe_lane_change_dist

    def act(self, *args, **kwargs):
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                self.set_target_speed(all_objects)
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        return [steering, acc]

    def set_target_speed(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        current_lane = self.control_object.lane
        total_lane_num = len(current_lanes)
        current_lane_idx = current_lane.index[-1]
        if current_lane_idx == 0 or current_lane_idx == current_lane_idx - 1:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        elif current_lane_idx % 2 == 0:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 3
        else:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST - 3
