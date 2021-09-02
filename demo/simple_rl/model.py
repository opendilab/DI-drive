import torch
import torch.nn as nn
from typing import List, Tuple, Union, Dict, Optional
from torch.distributions import Normal, Independent

from core.models import BEVSpeedConvEncoder
from core.models import BEVSpeedDeterminateNet, BEVSpeedStochasticNet, BEVSpeedSoftQNet, BEVSpeedProximalNet
from ding.model.common.head import DuelingHead, RegressionHead, ReparameterizationHead


class DQNRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: int = 15,
            hidden_dim_list: Tuple = [64, 128, 256],
            embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self._encoder = BEVSpeedConvEncoder(obs_shape, hidden_dim_list, embedding_dim, [3, 3, 3], [2, 2, 2])
        self._head = DuelingHead(embedding_dim, action_shape)

    def forward(self, obs):
        x = self._encoder(obs)
        y = self._head(x)
        return y


class DQNMultiDisRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Tuple = [3, 7],
            hidden_dim_list: Tuple = [64, 128, 256],
            embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self._encoder = BEVSpeedConvEncoder(obs_shape, hidden_dim_list, embedding_dim, [3, 3, 3], [2, 2, 2])
        self._head_acc = DuelingHead(embedding_dim, action_shape[0])
        self._head_steer = DuelingHead(embedding_dim, action_shape[1])

    def forward(self, obs):
        x = self._encoder(obs)
        acc_id = self._head_acc(x)['logit']
        steer_id = self._head_steer(x)['logit']
        return {'logit': [acc_id, steer_id]}


class DDPGRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_hidden_dim_list: List = [64, 128, 256],
            encoder_embedding_size: int = 512,
            obs_action_embedding_dim: int = 512,
            obs_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape

        self._actor = BEVSpeedDeterminateNet(
            self._obs_shape, self._act_shape, encoder_hidden_dim_list, encoder_embedding_size, obs_embedding_dim
        )
        self._critic = BEVSpeedDeterminateNet(
            self._obs_shape,
            self._act_shape,
            encoder_hidden_dim_list,
            encoder_embedding_size,
            obs_action_embedding_dim,
            is_critic=True
        )

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['compute_actor_critic', 'compute_actor', 'compute_critic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_critic(self, inputs: Dict) -> Dict:
        action = inputs['action']
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        q = self._critic(inputs['obs'], inputs['action'])
        return {'q_value': q}

    def compute_actor(self, inputs: Dict) -> Dict:
        obs = inputs
        action = self._actor(obs)
        return {'action': action}

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic


class TD3RLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_hidden_dim_list: List = [64, 128, 256],
            encoder_embedding_size: int = 512,
            obs_action_embedding_dim: int = 512,
            obs_embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape

        self._actor = BEVSpeedDeterminateNet(
            self._obs_shape, self._act_shape, encoder_hidden_dim_list, encoder_embedding_size, obs_embedding_dim
        )
        self._critic = nn.ModuleList(
            [
                BEVSpeedDeterminateNet(
                    self._obs_shape,
                    self._act_shape,
                    encoder_hidden_dim_list,
                    encoder_embedding_size,
                    obs_action_embedding_dim,
                    is_critic=True
                ) for _ in range(2)
            ]
        )

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['compute_actor_critic', 'compute_actor', 'compute_critic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_critic(self, inputs: Dict) -> Dict:
        action = inputs['action']
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        q = [self._critic[i](inputs['obs'], inputs['action']) for i in range(2)]
        return {'q_value': q}

    def compute_actor(self, inputs: Dict) -> Dict:
        obs = inputs
        action = self._actor(obs)
        return {'action': action}

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic


class SACRLModel(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_hidden_dim_list: List = [64, 128, 256],
            encoder_embedding_size: int = 512,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 512,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 512,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self._act = nn.ReLU()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self.twin_critic = twin_critic
        self.encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_dim_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])

        self.actor = nn.Sequential(
            nn.Linear(encoder_embedding_size, actor_head_hidden_size), activation,
            ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type='conditioned',
                activation=activation,
                norm_type=norm_type
            )
        )
        self.twin_critic = twin_critic
        if self.twin_critic:
            self._twin_encoder = BEVSpeedConvEncoder(self._obs_shape, encoder_hidden_dim_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2])
            self.critic = nn.ModuleList()
            for _ in range(2):
                self.critic.append(
                    nn.Sequential(
                        nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation,
                        RegressionHead(
                            critic_head_hidden_size,
                            1,
                            critic_head_layer_num,
                            final_tanh=False,
                            activation=activation,
                            norm_type=norm_type
                        )
                    )
                )
        else:
            self.critic = nn.Sequential(
                nn.Linear(encoder_embedding_size + self._act_shape, critic_head_hidden_size), activation,
                RegressionHead(
                    critic_head_hidden_size,
                    1,
                    critic_head_layer_num,
                    final_tanh=False,
                    activation=activation,
                    norm_type=norm_type
                )
            )

    def forward(self, inputs, mode=None, **kwargs):
        self.mode = ['compute_actor', 'compute_critic']
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x0 = self.encoder(inputs['obs'])
        x0 = torch.cat([x0, inputs['action']], dim=1)
        if self.twin_critic:
            x1 = self._twin_encoder(inputs['obs'])
            x1 = torch.cat([x1, inputs['action']], dim=1)
            x = [m(xi)['pred'] for m, xi in [(self.critic[0], x0), (self.critic[1], x1)]]
        else:
            x = self.critic(x0)['pred']
        return {'q_value': x}

    def compute_actor(self, inputs) -> Dict[str, torch.Tensor]:
        x = self.encoder(inputs)
        x = self.actor(x)
        return {'logit': [x['mu'], x['sigma']]}


class PPORLModel(nn.Module):
    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: int = 15,
            encoder_embedding_size: int = 512,
            encoder_hidden_dim_list: List = [64, 128, 256],
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self._actor = BEVSpeedProximalNet(
            self._obs_shape,
            self._act_shape,
            encoder_embedding_size,
            encoder_hidden_dim_list,
            is_critic=False
        )
        self._critic = BEVSpeedProximalNet(
            self._obs_shape,
            self._act_shape,
            encoder_embedding_size,
            encoder_hidden_dim_list,
            is_critic=True
        )

    def forward(self, inputs, mode=None, **kwargs):
        assert (mode in ['compute_actor_critic', 'compute_actor', 'compute_critic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def compute_actor_critic(self, inputs) -> Dict[str, torch.Tensor]:
        value = self._critic(inputs)
        logit = self._actor(inputs)
        return {'value': value, 'logit': logit}

    def compute_actor(self, inputs: Dict) -> Dict:
        obs = inputs
        logit = self._actor(obs)
        return {'logit': logit}

    def compute_critic(self, inputs: Dict) -> Dict:
        value = self._critic(inputs)
        return {'value': value}

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic
