import numpy as np
from typing import Any, Callable, Dict, Optional


class SteerNoiseWrapper(object):
    """
    Model wrapper to add noise in steer frequently.

    :Arguments:
        - model (Any): Model to wrap.
        - noise_type (str): Type to generate noise. Must in ['gauss', 'uniform'].
        - noise_kwargs (dict, optional): Arguments to set up noise generator. Defaults to {}.
        - noise_len (int, optional): Length of frames to apply noise. Defaults to 5.
        - drive_len (int, optional): Length of noise. Defaults to 100.
        - noise_range (dict, optional): Range of noise value, containing 'min' and 'max'. Defaults to None.

    :Instance: forward
    """

    def __init__(
            self,
            model: Any,
            noise_type: str = 'uniform',
            noise_args: dict = {},
            noise_len: int = 5,
            drive_len: int = 100,
            noise_range: Optional[dict] = None,
    ) -> None:
        self._model = model
        self._noise_func = get_noise_generator(noise_type, noise_args)
        self._noise_seq = {'drive': [drive_len, 'noise'], 'noise': [noise_len, 'drive']}
        self._noise_range = noise_range
        self._noise_state = 'drive'
        self._state_step = 0
        self._last_throttle = 0
        self._last_brake = 0

    def forward(self, *args, **kwargs) -> Dict:
        """
        Running forward and add noise.

        :Returns:
           Dict: Noised control signal with real control.
        """
        real_control = self._model.forward(*args, **kwargs)
        assert isinstance(real_control, dict), "model output must be dict, but find {}".format(type(real_control))

        control = {'brake': 0}
        for k, v in real_control.items():
            control['real_' + k] = v

        self._state_step += 1

        last_state = self._noise_state
        num_steps, next_state = self._noise_seq[self._noise_state]

        if self._noise_state == 'noise':
            control['steer'] = self._noise_steer
            control['throttle'] = self._last_throttle
            control['brake'] = self._last_brake
        else:
            control['steer'] = real_control['steer']
            control['throttle'] = real_control['throttle']
            control['brake'] = real_control['brake']
        control['noise_state'] = self._noise_state

        if self._state_step >= num_steps:
            self._state_step = 0
            self._noise_state = next_state
            self._noise_steer = self._noise_func()
            if self._noise_range is not None:
                self._noise_steer = np.clip(self._noise_steer, self._noise_range['min'], self._noise_range['max'])
            self._last_throttle = real_control['throttle']
            self._last_brake = real_control['brake']

        return control


def get_noise_generator(noise_type: str, noise_args: Dict) -> Callable:
    noise_dict = {
        'gauss': np.random.normal,
        'uniform': np.random.uniform,
    }
    if noise_type not in noise_dict:
        raise KeyError("not support noise type: {}".format(noise_type))
    else:
        noise_func = noise_dict[noise_type]
        return lambda: noise_func(**noise_args)
