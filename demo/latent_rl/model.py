import torch
import torch.nn as nn
from typing import List, Tuple, Union, Dict, Optional
from torch.distributions import Normal, Independent

from core.models import VanillaVAE
from ding.model.common import DiscreteHead, DuelingHead, MultiHead


class LatentDQNRLModel(nn.Module):
    def __init__(
            self,
            obs_shape: List = [192, 192, 7],
            action_shape: int = 100,
            latent_dim: int = 128,
            dueling: bool = True,
            head_hidden_size: Optional[int] = None,
            head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            vae_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        in_channels = obs_shape[-1]
        self._vae_model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dim)
        if vae_path is not None:
            state_dict = torch.load(vae_path)
            self._vae_model.load_state_dict(state_dict)

        if head_hidden_size is None:
            head_hidden_size = latent_dim + 12
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        multi_head = not isinstance(action_shape, int)
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    def forward(self, data: Dict) -> Dict:
        bev = data['birdview'].permute(0, 3, 1, 2)
        ego_info = data['ego_info']
        with torch.no_grad():
            mu, log_var = self._vae_model.encode(bev)
            feat = self._vae_model.reparameterize(mu, log_var)
        x = torch.cat([feat, ego_info], dim=1)
        x = self.head(x)
        return x
