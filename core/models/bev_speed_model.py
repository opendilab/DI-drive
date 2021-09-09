import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union

from ding.torch_utils import MLP


class BEVSpeedConvEncoder(nn.Module):
    """
    Convolutional encoder of Bird-eye View image and speed input. It takes a BeV image and a speed scalar as input.
    The BeV image is encoded by a convolutional encoder, to get a embedding feature which is half size of the
    embedding length. Then the speed value is repeated for half embedding length time, and concated to the above
    feature to get a final feature.

    :Arguments:
        - obs_shape (Tuple): BeV image shape.
        - hidden_dim_list (List): Conv encoder hidden layer dimension list.
        - embedding_size (int): Embedding feature dimensions.
        - kernel_size (List, optional): Conv kernel size for each layer. Defaults to [8, 4, 3].
        - stride (List, optional): Conv stride for each layer. Defaults to [4, 2, 1].
    """

    def __init__(
            self,
            obs_shape: Tuple,
            hidden_dim_list: List,
            embedding_size: int,
            kernel_size: List = [8, 4, 3],
            stride: List = [4, 2, 1],
    ) -> None:
        super().__init__()
        assert len(kernel_size) == len(stride), (kernel_size, stride)
        self._obs_shape = obs_shape
        self._embedding_size = embedding_size

        self._relu = nn.ReLU()
        layers = []
        input_dim = obs_shape[0]
        for i in range(len(hidden_dim_list)):
            layers.append(nn.Conv2d(input_dim, hidden_dim_list[i], kernel_size[i], stride[i]))
            layers.append(self._relu)
            input_dim = hidden_dim_list[i]
        layers.append(nn.Flatten())
        self._model = nn.Sequential(*layers)
        flatten_size = self._get_flatten_size()
        self._mid = nn.Linear(flatten_size, self._embedding_size // 2)

    def _get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self._model(test_data)
        return output.shape[1]

    def forward(self, data: Dict) -> torch.Tensor:
        """
        Forward computation of encoder

        :Arguments:
            - data (Dict): Input data, must contain 'birdview' and 'speed'

        :Returns:
            torch.Tensor: Embedding feature.
        """
        image = data['birdview'].permute(0, 3, 1, 2)
        speed = data['speed']
        x = self._model(image)
        x = self._mid(x)
        speed_embedding_size = self._embedding_size - self._embedding_size // 2
        speed_vec = torch.unsqueeze(speed, 1).repeat(1, speed_embedding_size)
        h = torch.cat((x, speed_vec), dim=1)
        return h


class FCContinuousNet(nn.Module):
    """
    Overview:
        FC continuous network which is used in ``QAC``.
        A main feature is that it uses ``_final_tanh`` to control whether
        add a tanh layer to scale the output to (-1, 1).
    Interface:
        __init__, forward
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            embedding_size: int = 64,
            final_tanh: bool = False,
            layer_num: int = 1,
    ) -> None:
        super(FCContinuousNet, self).__init__()
        self._act = nn.ReLU()
        self._main = nn.Sequential(
            MLP(input_size, embedding_size, embedding_size, layer_num + 1, activation=self._act),
            nn.Linear(embedding_size, output_size)
        )
        self._final_tanh = final_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._main(x)
        if self._final_tanh:
            x = torch.tanh(x)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x


class BEVSpeedDeterminateNet(nn.Module):
    """
    Actor Neural Network takes Bird-eye View image and speed and outputs actions determinately. It use a
    ``BEVSpeedConvEncoder`` to get a embedding feature, and use a fully-connected layer to get final output.
    It can be used as actor or critic network depending on forward arguments.

    :Arguments:
        - obs_shape (Tuple, optional): BeV image shape. Defaults to [5, 32, 32].
        - action_shape (Union[int, tuple], optional): Action shape. Defaults to 2.
        - encoder_hidden_dim_list (List, optional): Conv encoder hidden layer dimension list.
          Defaults to [64, 128, 256].
        - encoder_embedding_size (int, optional): Encoder output embedding size. Defaults to 512.
        - head_embedding_dim (int, optional): FC hidden layer dimension. Defaults to 512.
        - is_critic (bool, optional): Whether used as critic. Defaults to False.
    """

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_hidden_dim_list: List = [64, 128, 256],
            encoder_embedding_size: int = 512,
            head_embedding_dim: int = 512,
            is_critic: bool = False,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self._is_critic = is_critic

        self._encoder = BEVSpeedConvEncoder(
            self._obs_shape, encoder_hidden_dim_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
        )
        if is_critic:
            self._head = FCContinuousNet(encoder_embedding_size + self._act_shape, 1, head_embedding_dim)
        else:
            self._head = FCContinuousNet(encoder_embedding_size, self._act_shape, head_embedding_dim, final_tanh=True)

    def forward(self, obs: Dict, action: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward computation of network. If is critic, action must not be ``None``

        :Arguments:
            - obs (Dict): Observation dict.
            - action (Dict, optional): Action dict. Defaults to None.

        :Returns:
            torch.Tensor: Actions or critic value.
        """
        embedding = self._encoder(obs)
        if self._is_critic:
            assert action is not None
            obs_action_input = torch.cat([embedding, action], dim=1)
            q = self._head(obs_action_input)
            return q
        output = self._head(embedding)
        return output


class BEVSpeedStochasticNet(nn.Module):
    """
    Actor Neural Network takes Bird-eye View image and speed and outputs actions stochasticly. It use a
    ``BEVSpeedConvEncoder`` to get a embedding feature, and use a fully-connected layer to get mean and
    std values.

    :Arguments:
        - obs_shape (Tuple, optional): BeV image shape. Defaults to [5, 32, 32].
        - action_shape (Union[int, tuple], optional): Action shape. Defaults to 2.
        - encoder_hidden_dim_list (List, optional): Conv encoder hidden layer dimension list.
          Defaults to [64, 128, 256].
        - policy_hideen_size (int, optional): Encoder output embedding size. Defaults to 512.
        - log_std_min (int, optional): Log std min value. Defaults to -20.
        - log_std_max (int, optional): Log std max value. Defaults to 2.
        - init_w (float, optional): Clip value of mean and std layer weights. Defaults to 3e-3.
    """

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_hidden_dim_list: List = [64, 128, 256],
            policy_hideen_size: int = 512,
            log_std_min: int = -20,
            log_std_max: int = 2,
            init_w: float = 3e-3,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        self._encoder = BEVSpeedConvEncoder(
            self._obs_shape, encoder_hidden_dim_list, policy_hideen_size, [3, 3, 3], [2, 2, 2]
        )

        self._mean_layer = nn.Linear(policy_hideen_size, action_shape)
        self._mean_layer.weight.data.uniform_(-init_w, init_w)
        self._mean_layer.bias.data.uniform_(-init_w, init_w)

        self._log_std_layer = nn.Linear(policy_hideen_size, action_shape)
        self._log_std_layer.weight.data.uniform_(-init_w, init_w)
        self._log_std_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward computation of network.

        :Arguments:
            - obs (Dict): Observation dict.

        :Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and std value for actions.
        """
        embedding = self._encoder(obs)
        mean = self._mean_layer(embedding)
        log_std = self._log_std_layer(embedding)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        return mean, log_std


class BEVSpeedSoftQNet(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_hidden_dim_list: List = [64, 128, 256],
            soft_q_hidden_size: int = 512,
            init_w: float = 3e-3,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape

        self._encoder = BEVSpeedConvEncoder(
            self._obs_shape, encoder_hidden_dim_list, soft_q_hidden_size, [3, 3, 3], [2, 2, 2]
        )

        self._output_layer = nn.Linear(soft_q_hidden_size + self._act_shape, 1)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        embedding = self._encoder(obs)
        obs_action_input = torch.cat([embedding, action], dim=1)
        output = self._output_layer(obs_action_input)
        return output


class BEVSpeedProximalNet(nn.Module):

    def __init__(
            self,
            obs_shape: Tuple = [5, 32, 32],
            action_shape: Union[int, tuple] = 2,
            encoder_embedding_size: int = 512,
            encoder_hidden_dim_list: List = [64, 128, 256],
            head_hidden_size=128,
            head_layer_num=2,
            is_critic=False,
    ) -> None:
        super().__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self._encoder_embedding_size = encoder_embedding_size
        self._head_hidden_size = head_hidden_size
        self._head_layer_num = head_layer_num
        self._encoder = BEVSpeedConvEncoder(
            self._obs_shape, encoder_hidden_dim_list, encoder_embedding_size, [3, 3, 3], [2, 2, 2]
        )
        self._is_critic = is_critic
        if self._is_critic:
            self._head = self._setup_critic()
        else:
            self._head = self._setup_actor()

    def _setup_actor(self):
        if isinstance(self._act_shape, tuple):
            return nn.ModuleList([self._setup_1dim_actor(a) for a in self._act_shape])
        else:
            return self._setup_1dim_actor(self._act_shape)

    def _setup_critic(self):
        input_size = self._encoder_embedding_size
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_size, self._head_hidden_size))
            layers.append(nn.ReLU())
            input_size = self._head_hidden_size
        layers.append(nn.Linear(input_size, 1))
        output = nn.Sequential(*layers)
        return output

    def _setup_1dim_actor(self, act_shape: int) -> torch.nn.Module:
        input_size = self._encoder_embedding_size
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_size, self._head_hidden_size))
            layers.append(nn.ReLU())
            input_size = self._head_hidden_size
        layers.append(nn.Linear(input_size, act_shape))
        output = nn.Sequential(*layers)
        return output

    def forward(self, obs):
        embedding = self._encoder(obs)
        # Because we use the value AC, so the input of the head of actor and critic is the same form
        if self._is_critic:
            output = self._head(embedding)
        else:
            output = self._head(embedding)
        return output
