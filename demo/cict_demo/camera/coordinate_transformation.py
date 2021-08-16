import numpy as np


def rotationMatrix3D(roll, pitch, yaw):
    # RPY <--> XYZ, roll first, picth then, yaw final
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(3)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def rotationMatrixRoll(roll):
    R = np.identity(3)
    R[1, 1] = np.cos(roll)
    R[2, 2] = np.cos(roll)
    R[2, 1] = np.sin(roll)
    R[1, 2] = -np.sin(roll)
    return R


def rotarotationMatrixPitch(pitch):
    R = np.identity(3)
    R[0, 0] = np.cos(pitch)
    R[2, 2] = np.cos(pitch)
    R[2, 0] = -np.sin(pitch)
    R[0, 2] = np.sin(pitch)
    return R


def rotarotationMatrixYaw(yaw):
    R = np.identity(3)
    R[0, 0] = np.cos(yaw)
    R[1, 1] = np.cos(yaw)
    R[1, 0] = np.sin(yaw)
    R[0, 1] = -np.sin(yaw)
    return R


def rotationMatrix3DYPR(roll, pitch, yaw):
    return np.dot(np.dot(rotationMatrixRoll(roll), rotarotationMatrixPitch(pitch)), rotarotationMatrixYaw(yaw))


def reverseX():
    I = np.identity(3)
    I[0, 0] = -1
    return I


def reverseY():
    I = np.identity(3)
    I[1, 1] = -1
    return I


def intrinsicMatrix(fx, fy, u0, v0):
    K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    return K


class CoordinateTransformation(object):
    I = np.dot(np.dot(reverseX(), reverseY()), rotationMatrix3DYPR(np.pi / 2, 0, -np.pi / 2))

    @staticmethod
    def world3DToCamera3D(world_vec, R, t):
        camera_vec = np.dot(R.T, world_vec - t)
        return camera_vec

    @staticmethod
    def camera3DToWorld3D(camera_vec, R, t):
        world_vec = np.dot(R, camera_vec) + t
        return world_vec

    @staticmethod
    def camera3DToImage2D(camera_vec, K, eps=1e-24):
        image_vec = np.dot(np.dot(K, CoordinateTransformation.I), camera_vec)
        return image_vec[:2, :] / (image_vec[2, :] + eps)

    @staticmethod
    def world3DToImage2D(world_vec, K, R, t):
        camera_vec = CoordinateTransformation.world3DToCamera3D(world_vec, R, t)
        image_vec = CoordinateTransformation.camera3DToImage2D(camera_vec, K)
        return image_vec

    @staticmethod
    def world3DToImagePixel2D(world_vec, K, R, t):
        image_vec = CoordinateTransformation.world3DToImage2D(world_vec, K, R, t)
        x_pixel, y_pixel = round(image_vec[0, 0]), round(image_vec[1, 0])
        return np.array([x_pixel, y_pixel]).reshape(2, 1)

    @staticmethod
    def image2DToWorld3D(image_vec, K, R, t):
        r = np.vstack((image_vec, 1))
        b = np.vstack((np.dot(np.dot(K, CoordinateTransformation.I), t), 0))

        temp1 = np.dot(np.dot(K, CoordinateTransformation.I), R.T)
        temp2 = np.hstack((temp1, -r))
        A = np.vstack((temp2, np.array([[0, 0, 1, 0]])))
        world_vec = np.dot(np.linalg.inv(A), b)
        return world_vec[:3]

    @staticmethod
    def image2DToWorld3D2(image_vec, K, R, t):
        r = np.vstack((image_vec, np.ones((1, image_vec.shape[1]))))
        b = np.vstack((np.dot(np.dot(K, CoordinateTransformation.I), t), 0))

        temp1 = np.dot(np.dot(K, CoordinateTransformation.I), R.T)
        temp1 = np.expand_dims(temp1, axis=2).repeat(image_vec.shape[1], axis=2)
        r = np.expand_dims(r, axis=1)

        temp1 = np.transpose(temp1, (2, 0, 1))
        r = np.transpose(r, (2, 0, 1))

        temp2 = np.concatenate((temp1, -r), axis=2)
        temp3 = np.array([[0, 0, 1, 0]])
        temp3 = np.expand_dims(temp3, axis=2).repeat(image_vec.shape[1], axis=2)
        temp3 = np.transpose(temp3, (2, 0, 1))

        A = np.concatenate((temp2, temp3), axis=1)
        world_vec = np.dot(np.linalg.inv(A), b)
        return world_vec[:, :3]
