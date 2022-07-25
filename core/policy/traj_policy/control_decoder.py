import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent
from ding.rl_utils import create_noise_generator


class WpDecoder(nn.Module):

    def __init__(
        self,
        control_num=2,
        seq_len=30,
        use_relative_pos=True,
        dt=0.03,
        traj_control_mode='jerk',
    ):
        super(WpDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode
        # noise generator
        # noise_level = 0.0#np.arange(0.,1.0,0.1)
        # x0 = torch.zeros(1)
        # self.ou_pedal  = create_noise_generator(noise_type='ou',
        # noise_kwargs={'mu': 0, 'sigma': noise_level, 'theta': 0.15, 'x0': x0, 'dt':self.dt})
        # self.ou_steer = create_noise_generator(noise_type='ou',
        #  noise_kwargs={'mu': 0, 'sigma': noise_level, 'theta': 0.15, 'x0': x0, 'dt':self.dt})

    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt=0.03):
        # we assume the pedal batch and steer batch belongs to  [-1, 1]
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        # add OU noise to pedal and steer for DREX dataset
        # pedal_ou_noise = self.ou_pedal((1),pedal_batch.device)
        # steer_ou_noise = self.ou_steer((1),steering_batch.device)
        # pedal_batch += pedal_ou_noise
        # steering_batch += steer_ou_noise

        # Here we scale them to the fitable value
        steering_batch = steering_batch * 0.5
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        pedal_batch = pedal_batch * 5
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t

        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim=1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt=0.03):
        # x, y, theta, v, acc, steer,
        # control, jerk
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        pedal_t = prev_state[:, 4]
        steering_t = prev_state[:, 5]
        jerk_batch = jerk_batch * 4
        steering_rate_batch = steering_rate_batch * 0.5
        jerk_batch = torch.clamp(jerk_batch, -4, 4)
        steering_rate_batch = torch.clamp(steering_rate_batch, -0.5, 0.5)
        pedal_batch = pedal_t + jerk_batch * dt
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = steering_t + steering_rate_batch * dt
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 / 2, 3.14 / 2)
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t

        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1, pedal_batch, steering_batch], dim=1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state
        assert z.shape[1] == self.seq_len * 2
        for i in range(self.seq_len):
            control_1 = z[:, 2 * i]
            control_2 = z[:, 2 * i + 1]
            #curr_state = self.plant_model_batch(prev_state, jerk_batch, steer_rate_batch, self.dt)
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
            elif self.traj_control_mode == 'acc':
                curr_state = self.plant_model_acc(prev_state, control_1, control_2, self.dt)

            generated_traj.append(curr_state)
            prev_state = curr_state
        generated_traj = torch.stack(generated_traj, dim=1)
        return generated_traj

    def forward(self, z, init_state):
        return self.decode(z, init_state)


class CCDecoder(nn.Module):

    def __init__(
        self,
        control_num=2,
        seq_len=30,
        use_relative_pos=True,
        dt=0.03,
        traj_control_mode='jerk',
    ):
        super(CCDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.traj_control_mode = traj_control_mode

    # def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
    #     #import copy
    #     prev_state = prev_state_batch
    #     x_t = prev_state[:,0]
    #     y_t = prev_state[:,1]
    #     psi_t = prev_state[:,2]
    #     v_t = prev_state[:,3]
    #     steering_batch = steering_batch * 0.4
    #     #pedal_batch = torch.clamp(pedal_batch, -5, 5)
    #     steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
    #     beta = steering_batch
    #     a_t = pedal_batch * 4
    #     v_t_1 = v_t + a_t * dt
    #     v_t_1 = torch.clamp(v_t_1, 0, 10)
    #     psi_dot = v_t * torch.tan(beta) / 2.5
    #     psi_t_1 = psi_dot*dt + psi_t
    #     x_dot = v_t_1 * torch.cos(psi_t_1)
    #     y_dot = v_t_1 * torch.sin(psi_t_1)
    #     x_t_1 = x_dot * dt + x_t
    #     y_t_1 = y_dot * dt + y_t

    #     #psi_t = self.wrap_angle_rad(psi_t)
    #     current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
    #     #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
    #     return current_state

    def plant_model_acc(self, prev_state_batch, pedal_batch, steering_batch, dt=0.03):
        # we assume the pedal batch and steer batch belongs to  [-1, 1]
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        # Here we scale them to the fitable value
        steering_batch = steering_batch * 0.5
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        pedal_batch = pedal_batch * 5
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t

        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim=1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def plant_model_jerk(self, prev_state_batch, jerk_batch, steering_rate_batch, dt=0.03):
        # x, y, theta, v, acc, steer,
        # control, jerk
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:, 0]
        y_t = prev_state[:, 1]
        psi_t = prev_state[:, 2]
        v_t = prev_state[:, 3]
        pedal_t = prev_state[:, 4]
        steering_t = prev_state[:, 5]
        jerk_batch = jerk_batch * 4
        steering_rate_batch = steering_rate_batch * 0.5
        jerk_batch = torch.clamp(jerk_batch, -4, 4)
        steering_rate_batch = torch.clamp(steering_rate_batch, -0.5, 0.5)
        pedal_batch = pedal_t + jerk_batch * dt
        pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = steering_t + steering_rate_batch * dt
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 / 2, 3.14 / 2)
        psi_t_1 = psi_dot * dt + psi_t
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t
        y_t_1 = y_dot * dt + y_t

        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1, pedal_batch, steering_batch], dim=1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state
        assert z.shape[1] == 2
        for i in range(self.seq_len):
            control_1 = z[:, 0]
            control_2 = z[:, 1]
            if self.traj_control_mode == 'jerk':
                curr_state = self.plant_model_jerk(prev_state, control_1, control_2, self.dt)
            elif self.traj_control_mode == 'acc':
                curr_state = self.plant_model_acc(prev_state, control_1, control_2, self.dt)
            #curr_state = self.plant_model_batch(prev_state, pedal_batch, steer_batch, self.dt)
            generated_traj.append(curr_state)
            prev_state = curr_state
        generated_traj = torch.stack(generated_traj, dim=1)
        return generated_traj

    def forward(self, z, init_state):
        return self.decode(z, init_state)
