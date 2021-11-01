import numpy as np
import copy
import cv2
import os
import PIL.Image as Image
from PIL import ImageDraw
import lmdb
from tqdm import tqdm
import carla
from collect_pm import CollectPerspectiveImage, InversePerspectiveMapping

config = dict(
    env=dict(
        env_num=5,
        simulator=dict(
            disable_two_wheels=True,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[640, 360],
                    position=[0.5, 0.0, 2.5],
                    rotation=[0, 0, 0],
                    sensor_tick=1. / 30,
                ),
                dict(
                    name='lidar',
                    type='lidar',
                    channels=64,
                    range=50,
                    points_per_second=100000,
                    rotation_frequency=30,
                    upper_fov=10,
                    lower_fov=-30,
                    position=[0.5, 0.0, 2.5],
                    rotation=[0, 0, 0],
                    sensor_tick=0.05,
                )
            ),
            verbose=True,
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
        wrapper=dict(suite='FullTown01-v3', ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9010, 2]),
    ],
    policy=dict(
        target_speed=25,
        noise=False,
        collect=dict(
            n_episode=5,
            dir_path='datasets/cict_datasets_train',
            npy_prefix='_preloads',
            collector=dict(suite='FullTown01-v3', ),
        ),
    ),
)

scale = 12.0
x_offset = 800
y_offset = 1000

MAX_SPEED = 50
TRAJ_LENGTH = 25
MIN_TRAJ_LENGTH = 15

vehicle_width = 2.0
longitudinal_sample_number_near = 8
longitudinal_sample_number_far = 0.5
longitudinal_length = 25.0
lateral_sample_number = 20
lateral_step_factor = 1.0
ksize = 21

sensor_config = {'rgb': {'img_height': 360, 'img_width': 640, 'fov': 120, 'location': [0.5, 0, 2.5]}}


class Param(object):

    def __init__(self):
        self.traj_length = float(TRAJ_LENGTH)
        self.target_speed = float(MAX_SPEED)
        self.vehicle_width = float(vehicle_width)
        self.longitudinal_sample_number_near = longitudinal_sample_number_near
        self.longitudinal_sample_number_far = longitudinal_sample_number_far
        self.lateral_sample_number = lateral_sample_number
        self.lateral_step_factor = lateral_step_factor
        self.longitudinal_length = longitudinal_length
        self.ksize = ksize
        self.sensor_config = sensor_config


params = Param()


def get_map():
    origin_map = np.zeros((6000, 6000, 3), dtype="uint8")
    #origin_map.fill(255)
    origin_map = Image.fromarray(origin_map)
    return origin_map


def draw_point(waypoint_list, origin_map):
    route_list = []
    for waypoint in waypoint_list:
        x = scale * waypoint[0] + x_offset
        y = scale * waypoint[1] + y_offset
        route_list.append(x)
        route_list.append(y)

    draw = ImageDraw.Draw(origin_map)
    draw.point(route_list, fill=(255, 255, 255))
    #print(route_list)
    #print(waypoint_list)
    return origin_map


def draw_route(waypoint_list, origin_map):

    route_list = []
    for waypoint in waypoint_list:
        x = scale * waypoint[0] + x_offset
        y = scale * waypoint[1] + y_offset
        route_list.append(x)
        route_list.append(y)

    draw = ImageDraw.Draw(origin_map)
    draw.line(route_list, 'red', width=30)
    #print(route_list)
    #print(waypoint_list)
    return origin_map


def find_dest_with_fix_length(start, waypoint_list):
    length = start
    for i in range(len(waypoint_list) - 1):
        length += np.linalg.norm(waypoint_list[i + 1][:2] - waypoint_list[i][:2])
        if length >= params.traj_length:
            return waypoint_list[i + 1][:2], i + 1
    return waypoint_list[-1][:2], -1


def draw_destination(location, waypoint_list, origin_map):
    start = np.linalg.norm(waypoint_list[0][:2] - location[:2])
    #print(location, waypoint_list[0], start)
    dest, _ = find_dest_with_fix_length(start, waypoint_list)

    x = scale * dest[0] + x_offset
    y = scale * dest[1] + y_offset
    #print(dest, x, y)

    draw = ImageDraw.Draw(origin_map)
    draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill='red', outline='red', width=30)
    return origin_map


def get_nav(location, rotation, plan_map, town=1):
    if town == 1:
        x_offset = 800
        y_offset = 1000
    elif town == 2:
        x_offset = 1500
        y_offset = 0

    x = int(scale * location[0] + x_offset)
    y = int(scale * location[1] + y_offset)
    #print(x, y, plan_map)
    _nav = plan_map.crop((x - 400, y - 400, x + 400, y + 400))
    im_rotate = _nav.rotate(rotation[1] + 90)
    nav = im_rotate.crop((_nav.size[0] // 2 - 320, _nav.size[1] // 2 - 360, _nav.size[0] // 2 + 320, _nav.size[1] // 2))
    #print(nav)
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)

    return nav


'''
def get_bezier(location, waypoint_list):
    total_length = [np.linalg.norm(location[:2] - waypoint_list[0][:2])]

    for i in range(len(waypoint_list)-1):
        total_length.append(np.linalg.norm(waypoint_list[i][:2] - waypoint_list[i+1][:2]) + total_length[-1])

    t = np.array(total_length[:-1]).reshape(-1, 1) / total_length[-1]
    b0 = location[:2].reshape(1, 2)
    b4 = waypoint_list[-1][:2].reshape(1, 2)
    B0 = (1 - t) ** 4
    B4 = t ** 4
    p = waypoint_list[:-1, :2] - np.dot(np.concatenate([B0, B4], axis=1), np.concatenate([b0, b4], axis=0))
    B1 = 4 * t * ((1 - t) ** 3)
    B2 = 6 * (t ** 2) * ((1 - t) ** 2)
    B3 = 4 * (1 - t) * (t ** 3)
    Bm = np.concatenate([B1, B2, B3], axis=1)
    bm = np.dot(np.linalg.inv(np.dot(Bm.T, Bm)), Bm.T)
    bm = np.dot(bm, p)
    b = np.concatenate([b0, bm, b4], axis=0)
    t = np.linspace(0, 1, 100)
    t = t.reshape(100, 1)
    B0 = (1 - t) ** 4
    B4 = t ** 4
    B1 = 4 * t * ((1 - t) ** 3)
    B2 = 6 * (t ** 2) * ((1 - t) ** 2)
    B3 = 4 * (1 - t) * (t ** 3)
    B = np.concatenate([B0, B1, B2, B3, B4], axis=1)
    bezier_list = np.dot(B, b)
    #print(b)
    return bezier_list, b
'''


def destination(save_dir, episode_path):
    lmdb_file = lmdb.open(os.path.join(save_dir, episode_path, 'measurements.lmdb')).begin()
    waypoint_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path)) if (x.endswith('npy') and x.startswith('way'))
    ]

    waypoint_file.sort()
    #print(waypoint_file)
    for k in tqdm(waypoint_file):
        index = k.split('_')[1].split('.')[0]
        measurements = np.frombuffer(lmdb_file.get(('measurements_%05d' % int(index)).encode()), np.float32)
        location = np.array([measurements[7], measurements[8], measurements[9]]).astype(np.float32)
        rotation = np.array([measurements[18], measurements[19], measurements[20]]).astype(np.float32)
        waypoint_list = np.load(os.path.join(save_dir, episode_path, k))
        origin_map = get_map()
        plan_map = draw_destination(location, waypoint_list, copy.deepcopy(origin_map))
        dest = get_nav(location, rotation, plan_map, town=1)
        cv2.imwrite(os.path.join(save_dir, episode_path, 'dest_%05d.png' % int(index)), dest)


class Sensor(object):

    def __init__(self, config):
        self.type_id = 'sensor.camera.rgb'
        self.transform = carla.Transform(
            carla.Location(x=config['location'][0], y=config['location'][1], z=config['location'][2])
        )
        self.attributes = dict()
        self.attributes['role_name'] = 'front'
        self.attributes['image_size_x'] = str(config['img_width'])
        self.attributes['image_size_y'] = str(config['img_height'])
        self.attributes['fov'] = str(config['fov'])

    def get_transform(self):
        return self.transform


def find_traj_with_fix_length(start_index, pose_list):
    length = 0.0
    for i in range(start_index, len(pose_list) - 1):
        length += pose_list[i].location.distance(pose_list[i + 1].location)
        if length >= params.traj_length:
            return i + 1
    return -1


def destination2(save_dir, episode_path):
    lmdb_file = lmdb.open(os.path.join(save_dir, episode_path, 'measurements.lmdb')).begin()
    waypoint_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path)) if (x.endswith('npy') and x.startswith('way'))
    ]

    waypoint_file.sort()
    #print(waypoint_file)
    sensor = Sensor(params.sensor_config['rgb'])
    collect_perspective = CollectPerspectiveImage(params, sensor)
    commands = []
    for k in tqdm(waypoint_file):
        index = k.split('_')[1].split('.')[0]
        measurements = np.frombuffer(lmdb_file.get(('measurements_%05d' % int(index)).encode()), np.float32)
        location = np.array([measurements[7], measurements[8], measurements[9]]).astype(np.float32)
        rotation = np.array([measurements[18], measurements[19], measurements[20]]).astype(np.float32)
        waypoint_list = np.load(os.path.join(save_dir, episode_path, k))
        start = np.linalg.norm(waypoint_list[0][:2] - location[:2])
        #print(location, waypoint_list[0], start)
        dest, _ = find_dest_with_fix_length(start, waypoint_list)
        #if location[0] - dest[0] > 0.5:
        #    commands.append(1)
        #elif location[0] - dest[0] < -0.5:
        #    commands.append(2)
        #else:
        #    commands.append(0)
        zero = np.zeros((3, 1))
        zero[:2, 0] = dest
        dest_map = collect_perspective.drawDestInImage(zero, location, rotation)
        cv2.imwrite(os.path.join(save_dir, episode_path, 'dest2_%05d.png' % int(index)), dest_map)

    #np.save(os.path.join(save_dir, episode_path, 'commands.npy'), np.array(commands))


def get_potential_map(save_dir, episode_path, measurements, img_file):
    pm_dir = os.path.join(save_dir, episode_path, 'pm')
    print(pm_dir)
    if not os.path.exists(pm_dir):
        os.mkdir(pm_dir)

    pose_list = []
    loc_list = []
    for measurement in measurements:

        transform = carla.Transform()
        transform.location.x = float(measurement['location'][0])
        transform.location.y = float(measurement['location'][1])
        transform.location.z = float(measurement['location'][2])
        transform.rotation.pitch = float(measurement['rotation'][0])
        transform.rotation.yaw = float(measurement['rotation'][1])
        transform.rotation.roll = float(measurement['rotation'][2])
        pose_list.append(transform)
        loc_list.append(measurement['location'])

    sensor = Sensor(params.sensor_config['rgb'])
    collect_perspective = CollectPerspectiveImage(params, sensor)

    for index in tqdm(range(len(pose_list))):
        end_index = find_traj_with_fix_length(index, pose_list)
        if end_index < 0:
            print('no enough traj: ', index, index / len(pose_list))
            break

        vehicle_transform = pose_list[index]  # in world coordinate
        traj_pose_list = []
        traj_list = []
        for i in range(index, end_index):
            traj_pose_list.append((i, pose_list[i]))
            traj_list.append(loc_list[i])

        #t1 = time.time()
        '''
        bezier_list, bezier_coff = get_bezier(traj_list[0], np.stack(traj_list[1:], axis=0))
        measurements[index]['bezier_coff'] = bezier_coff
        origin_map = get_map()
        plan_map = draw_route(bezier_list, copy.deepcopy(origin_map))
        plan_map = draw_point(traj_list, plan_map)
        nav = get_nav(measurements[index]['location'], measurements[index]['rotation'], plan_map, town=1)
        cv2.imwrite(os.path.join(save_dir, episode_path, 'nav_%05d.png' % int(index)), nav)
        '''

        empty_image = collect_perspective.getPM(traj_pose_list, vehicle_transform)
        #t2 = time.time()

        #cv2.imshow('empty_image', empty_image)
        #cv2.waitKey(3)
        cv2.imwrite(os.path.join(pm_dir, '%05d.png' % index), empty_image)

    return measurements


def get_inverse_potential_map(save_dir, episode_path, pm_file, lidar_file):
    ipm_dir = os.path.join(save_dir, episode_path, 'ipm')
    if not os.path.exists(ipm_dir):
        os.mkdir(ipm_dir)

    sensor = Sensor(params.sensor_config['rgb'])
    inverse_perspective_mapping = InversePerspectiveMapping(params, sensor)

    for i in tqdm(range(len(pm_file))):
        pm = cv2.imread(os.path.join(save_dir, pm_file[i]))
        lidar = np.load(os.path.join(save_dir, lidar_file[i]))

        ipm = inverse_perspective_mapping.getIPM(pm)
        img = inverse_perspective_mapping.get_cost_map(ipm, lidar)
        cv2.imwrite(os.path.join(ipm_dir, '%05d.png' % i), img)


def get_option(option_name, end_ind):
    x = np.load(option_name, allow_pickle=True)
    option = x[0] - 1
    end_ind = len(option_name) if end_ind == -1 else end_ind + 1
    for o in x[1:end_ind]:
        if o != 4:
            option = o - 1
            break
    return option


def save_as_npy(save_dir, episode_path):

    lmdb_file = lmdb.open(os.path.join(save_dir, episode_path, 'measurements.lmdb')).begin()
    dest_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path)) if (x.endswith('png') and x.startswith('dest_'))
    ]
    dest_file.sort()
    dest_file2 = [
        x for x in os.listdir(os.path.join(save_dir, episode_path)) if (x.endswith('png') and x.startswith('dest2_'))
    ]
    dest_file2.sort()
    img_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path)) if (x.endswith('png') and x.startswith('rgb'))
    ]
    img_file.sort()
    #print(waypoint_file)
    measurements_list = []

    for k in tqdm(img_file):
        index = k.split('_')[1].split('.')[0]
        measurements = np.frombuffer(lmdb_file.get(('measurements_%05d' % int(index)).encode()), np.float32)
        data = {}
        data['time'] = float(measurements[1])
        data['acceleration'] = np.array([measurements[4], measurements[5], measurements[6]], dtype=np.float32)
        data['location'] = np.array([measurements[7], measurements[8], measurements[9]], dtype=np.float32)
        data['direction'] = float(measurements[11]) - 1.
        data['velocity'] = np.array([measurements[12], measurements[13], measurements[14]], dtype=np.float32)
        data['angular_velocity'] = np.array([measurements[15], measurements[16], measurements[17]], dtype=np.float32)
        data['rotation'] = np.array([measurements[18], measurements[19], measurements[20]]).astype(np.float32)
        data['steer'] = float(measurements[21])
        data['throttle'] = float(measurements[22])
        data['brake'] = float(measurements[23])
        data['real_steer'] = float(measurements[24])
        data['real_throttle'] = float(measurements[25])
        data['real_brake'] = float(measurements[26])
        data['tl_state'] = float(measurements[27])
        data['tl_distance'] = float(measurements[28])

        waypoint_list = np.load(os.path.join(save_dir, episode_path, 'waypoints_%05d.npy' % int(index)))

        start = np.linalg.norm(data['location'][:2] - waypoint_list[0][:2])
        _, end_ind = find_dest_with_fix_length(start, waypoint_list)
        data['option'] = get_option(os.path.join(save_dir, episode_path, 'direction_%05d.npy' %
                                                 int(index)), end_ind) if data['direction'] == 3 else data['direction']
        #print(episode_path, int(index), data['option'], data['command'])

        measurements_list.append(data)

    dest_file = [os.path.join(episode_path, x) for x in dest_file]
    dest_file2 = [os.path.join(episode_path, x) for x in dest_file2]
    img_file = [os.path.join(episode_path, x) for x in img_file]

    measurements_list = get_potential_map(save_dir, episode_path, measurements_list, img_file)

    pm_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path, 'pm'))
        if x.endswith('png') and (not x.startswith('fake'))
    ]
    pm_file.sort()
    pm_file = [os.path.join(episode_path, 'pm', x) for x in pm_file]

    lidar_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path)) if (x.endswith('npy') and x.startswith('lidar'))
    ]
    lidar_file.sort()
    lidar_file = [os.path.join(episode_path, x) for x in lidar_file]
    get_inverse_potential_map(save_dir, episode_path, pm_file, lidar_file)
    ipm_file = [
        x for x in os.listdir(os.path.join(save_dir, episode_path, 'ipm'))
        if x.endswith('png') and (not x.startswith('pred'))
    ]
    ipm_file.sort()
    ipm_file = [os.path.join(episode_path, 'ipm', x) for x in ipm_file]

    if not os.path.exists(config['policy']['collect']['npy_prefix']):
        os.mkdir(config['policy']['collect']['npy_prefix'])
    np.save(
        '%s/%s.npy' % (config['policy']['collect']['npy_prefix'], episode_path),
        [img_file, dest_file, dest_file2, pm_file, ipm_file, measurements_list]
    )


if __name__ == '__main__':
    save_dir = config['policy']['collect']['dir_path']
    epi_folder = [x for x in os.listdir(save_dir) if x.startswith('epi')]
    #epi_folder = ['episode_00038','episode_00039']
    #epi_folder = ['episode_00037']

    for episode_path in tqdm(epi_folder):
        destination(save_dir, episode_path)
        destination2(save_dir, episode_path)
        save_as_npy(save_dir, episode_path)
