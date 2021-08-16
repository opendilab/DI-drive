import numpy as np

from .coordinate_transformation import CoordinateTransformation, rotationMatrix3D, intrinsicMatrix


class IntrinsicParams(object):

    def __init__(self, sensor):
        '''
        Args:
            sensor: carla.Sensor
        '''
        image_size_x = float(sensor.attributes['image_size_x'])
        image_size_y = float(sensor.attributes['image_size_y'])
        fov = eval(sensor.attributes['fov'])
        f = image_size_x / (2 * np.tan(fov * np.pi / 360))

        # [px]
        self.fx = f
        self.fy = f
        self.u0 = image_size_x / 2
        self.v0 = image_size_y / 2


class ExtrinsicParams(object):

    def __init__(self, sensor):
        '''
        Args:
            sensor: carla.Sensor
        '''

        # camera coordinate in world coordinate
        transform = sensor.get_transform()

        # [m]
        self.x = transform.location.x
        self.y = transform.location.y
        self.z = transform.location.z

        # [rad]
        self.roll = np.deg2rad(transform.rotation.roll)
        self.pitch = np.deg2rad(transform.rotation.pitch)
        self.yaw = np.deg2rad(transform.rotation.yaw)


class CameraParams(object):

    def __init__(self, intrinsic_params, extrinsic_params):
        '''
        Args:
            intrinsic_params: IntrinsicParams
            extrinsic_params: ExtrinsicParams
        '''

        self.K = intrinsicMatrix(intrinsic_params.fx, intrinsic_params.fy, intrinsic_params.u0, intrinsic_params.v0)

        # (coordinate) t: camera in world, R: camera to world
        self.t = np.array([[extrinsic_params.x, extrinsic_params.y, extrinsic_params.z]]).T
        self.R = rotationMatrix3D(extrinsic_params.roll, extrinsic_params.pitch, extrinsic_params.yaw)
