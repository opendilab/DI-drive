from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np
from torch.distributions import Independent, Normal

from ding.torch_utils import Adam, to_device
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, get_gae_with_default_last_value, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample, gae, gae_data, ppo_error_continuous, \
    get_gae
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from ding.policy import Policy
from ding.policy import PPOPolicy

from ding.policy.common_utils import default_preprocess_learn
from ding.utils import dicts_to_lists, lists_to_dicts


@POLICY_REGISTRY.register('traj_ppo')
class TrajPPO(PPOPolicy):

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:  # new v network compute new value
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], data['traj_flag'])
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                    unnormalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = unnormalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = unnormalized_returns

            else:  # don't recompute adv
                if self._value_norm:
                    unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = unnormalized_return / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                adv = batch['adv']
                if self._adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Calculate ppo error
                if self._action_space == 'continuous':
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['latent_action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error_continuous(ppo_batch, self._clip_ratio)
                elif self._action_space == 'discrete':
                    ppo_batch = ppo_data(
                        output['logit'], batch['logit'], batch['latent_action'], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)
                elif self._action_space == 'hybrid':
                    # discrete part (discrete policy loss and entropy loss)
                    ppo_discrete_batch = ppo_policy_data(
                        output['logit']['action_type'], batch['logit']['action_type'],
                        batch['latent_action']['action_type'], adv, batch['weight']
                    )
                    ppo_discrete_loss, ppo_discrete_info = ppo_policy_error(ppo_discrete_batch, self._clip_ratio)
                    # continuous part (continuous policy loss and entropy loss, value loss)
                    ppo_continuous_batch = ppo_data(
                        output['logit']['action_args'], batch['logit']['action_args'], batch['action']['action_args'],
                        output['value'], batch['value'], adv, batch['return'], batch['weight']
                    )
                    ppo_continuous_loss, ppo_continuous_info = ppo_error_continuous(
                        ppo_continuous_batch, self._clip_ratio
                    )
                    # sum discrete and continuous loss
                    ppo_loss = type(ppo_continuous_loss)(
                        ppo_continuous_loss.policy_loss + ppo_discrete_loss.policy_loss, ppo_continuous_loss.value_loss,
                        ppo_continuous_loss.entropy_loss + ppo_discrete_loss.entropy_loss
                    )
                    ppo_info = type(ppo_continuous_info)(
                        max(ppo_continuous_info.approx_kl, ppo_discrete_info.approx_kl),
                        max(ppo_continuous_info.clipfrac, ppo_discrete_info.clipfrac)
                    )
                wv, we = self._value_weight, self._entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                }
                if self._action_space == 'continuous':
                    return_info.update(
                        {
                            'act': batch['action'].float().mean().item(),
                            'mu_mean': output['logit']['mu'].mean().item(),
                            'sigma_mean': output['logit']['sigma'].mean().item(),
                        }
                    )
                return_infos.append(return_info)
        return return_infos

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        init_state = data['vehicle_state']
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic')
            latent_action = output['action']
            output['latent_action'] = latent_action
            traj = self._collect_model.generate_traj_from_lat(output['latent_action'], init_state)
            output['trajectory'] = traj
            output['action'] = traj
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - model_output (:obj:`dict`): Output of collect model, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'value': model_output['value'],
            'latent_action': model_output['latent_action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        init_state = data['vehicle_state']
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
            latent_action = output['action']
            output['latent_action'] = latent_action
            traj = self._collect_model.generate_traj_from_lat(output['latent_action'], init_state)
            output['trajectory'] = traj
            output['action'] = traj
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = to_device(data, self._device)
        for transition in data:
            transition['traj_flag'] = copy.deepcopy(transition['done'])
        data[-1]['traj_flag'] = True

        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            if self._cfg.multi_agent:
                with torch.no_grad():
                    last_value = self._collect_model.forward(
                        {
                            'agent_state': data[-1]['next_obs']['agent_state'].unsqueeze(0),
                            'global_state': data[-1]['next_obs']['global_state'].unsqueeze(0),
                            'action_mask': data[-1]['next_obs']['action_mask'].unsqueeze(0)
                        },
                        mode='compute_actor_critic'
                    )['value']
                    last_value = last_value.squeeze(0)
            else:
                with torch.no_grad():
                    last_value = self._collect_model.forward(data[-1]['next_obs'], mode='compute_actor_critic')['value']
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        # remove next_obs for save memory when not recompute adv
        if not self._recompute_adv:
            for i in range(len(data)):
                data[i].pop('next_obs')
        return get_train_sample(data, self._unroll_len)
