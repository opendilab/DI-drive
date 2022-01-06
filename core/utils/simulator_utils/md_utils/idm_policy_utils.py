from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
import logging


class MacroIDMPolicy(IDMPolicy):

    def __init__(self, control_object, random_seed):
        super(MacroIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 50
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST

    # def lane_change_policy(self, all_objects):
    #     current_lanes = self.control_object.navigation.current_ref_lanes
    #     surrounding_objects = FrontBackObjects.get_find_front_back_objs(
    #         all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
    #     )
    #     if not surrounding_objects.right_lane_exist():
    #         self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 5
    #     elif not surrounding_objects.left_lane_exist():
    #         self.NORMAL_SPEED = self.NORMAL_SPEED_CONST - 5
    #     IDMPolicy.lane_change_policy(self, all_objects)

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
        if not surrounding_objects.right_lane_exist():
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + 5
        elif not surrounding_objects.left_lane_exist():
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST - 5
        else:
            self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
