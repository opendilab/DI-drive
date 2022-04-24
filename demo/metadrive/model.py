from torch import nn
from typing import Union, Dict, Optional, List
from easydict import EasyDict
import torch

from ding.utils import SequenceType, squeeze
from ding.model.template import QAC, VAC
from ding.model.common import RegressionHead, ReparameterizationHead, FCEncoder, DiscreteHead, MultiHead
from core.models.common_model import ConvEncoder


class ConvQAC(QAC):

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str,
        encoder_hidden_size_list: SequenceType = [64],
        twin_critic: bool = False,
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
    ):
        super(QAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )

        self.action_shape = action_shape
        self.action_space = action_space
        assert self.action_space in ['regression', 'reparameterization', 'hybrid']
        if self.action_space == 'regression':  # DDPG, TD3
            self.actor = nn.Sequential(
                encoder_cls(obs_shape, encoder_hidden_size_list, activation=None, norm_type=norm_type), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.action_space == 'reparameterization':  # SAC
            self.actor = nn.Sequential(
                encoder_cls(obs_shape, encoder_hidden_size_list, activation=None, norm_type=norm_type), activation,
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
            self.critic_encoder = nn.ModuleList()
            self.critic_head = nn.ModuleList()
            for _ in range(2):
                self.critic_encoder.append(
                    encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
                )
                self.critic_head.append(
                    RegressionHead(
                        critic_head_hidden_size + action_shape,
                        1,
                        critic_head_layer_num,
                        final_tanh=False,
                        activation=activation,
                        norm_type=norm_type
                    )
                )
        else:
            self.critic_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
            self.critic_head = RegressionHead(
                critic_head_hidden_size + action_shape,
                1,
                critic_head_layer_num,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type
            )
        if self.twin_critic:
            self.critic = nn.ModuleList([*self.critic_encoder, *self.critic_head])
        else:
            self.critic = nn.ModuleList([self.critic_encoder, self.critic_head])

    def compute_critic(self, inputs: Dict) -> Dict:
        if self.twin_critic:
            x = [m(inputs['obs']) for m in self.critic_encoder]
            x = [torch.cat([x1, inputs['action']], dim=1) for x1 in x]
            x = [m(xi)['pred'] for m, xi in [(self.critic_head[0], x[0]), (self.critic_head[1], x[1])]]
        else:
            x = self.critic_encoder(inputs['obs'])
            x = self.critic_head(x)['pred']
        return {'q_value': x}


class ConvVAC(VAC):
    r"""
    Overview:
        The VAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str = 'discrete',
        share_encoder: bool = True,
        encoder_hidden_size_list: SequenceType = [128, 128, 64],
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        fixed_sigma_value: Optional[int] = 0.3,
        bound_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the VAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - action_space (:obj:`str`): Choose action head in ['discrete', 'continuous', 'hybrid']
            - share_encoder (:obj:`bool`): Whether share encoder.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(VAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        # Encoder Type
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        self.share_encoder = share_encoder
        if self.share_encoder:
            self.encoder = encoder_cls(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            self.actor_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
            self.critic_encoder = encoder_cls(
                obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        # Head Type
        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )
        self.action_space = action_space
        assert self.action_space in ['discrete', 'continuous', 'hybrid'], self.action_space
        if self.action_space == 'continuous':
            self.multi_head = False
            self.actor_head = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type
            )
        elif self.action_space == 'discrete':
            actor_head_cls = DiscreteHead
            multi_head = not isinstance(action_shape, int)
            self.multi_head = multi_head
            if multi_head:
                self.actor_head = MultiHead(
                    actor_head_cls,
                    actor_head_hidden_size,
                    action_shape,
                    layer_num=actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            else:
                self.actor_head = actor_head_cls(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
        elif self.action_space == 'hybrid':  # HPPO
            # hybrid action space: action_type(discrete) + action_args(continuous),
            # such as {'action_type_shape': torch.LongTensor([0]), 'action_args_shape': torch.FloatTensor([0.1, -0.27])}
            action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
            action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
            actor_action_args = ReparameterizationHead(
                actor_head_hidden_size,
                action_shape.action_args_shape,
                actor_head_layer_num,
                sigma_type=sigma_type,
                fixed_sigma_value=fixed_sigma_value,
                activation=activation,
                norm_type=norm_type,
                bound_type=bound_type,
            )
            actor_action_type = DiscreteHead(
                actor_head_hidden_size,
                action_shape.action_type_shape,
                actor_head_layer_num,
                activation=activation,
                norm_type=norm_type,
            )
            self.actor_head = nn.ModuleList([actor_action_type, actor_action_args])

        # must use list, not nn.ModuleList
        if self.share_encoder:
            self.actor = [self.encoder, self.actor_head]
            self.critic = [self.encoder, self.critic_head]
        else:
            self.actor = [self.actor_encoder, self.actor_head]
            self.critic = [self.critic_encoder, self.critic_head]
        # Convenient for calling some apis (e.g. self.critic.parameters()),
        # but may cause misunderstanding when `print(self)`
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)
