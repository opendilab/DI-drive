import importlib

import numpy as np
import scipy
import scipy.misc
import torch
import torch.nn as nn

from core.utils.model_utils.common import FC, Conv, Branching, Join
from core.utils.others.general_helper import command_number_to_index


class CoILICRA(nn.Module):

    def __init__(self, cfg):
        # TODO: Improve the model autonaming function

        super(CoILICRA, self).__init__()
        params = cfg.MODEL_CONFIGURATION
        self.params = params

        number_first_layer_channels = 0

        for _, sizes in cfg.SENSORS.items():
            number_first_layer_channels += sizes[0] * cfg.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(cfg.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1], sensor_input_shape[2]]
        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in params['perception']:
            perception_convs = Conv(
                params={
                    'channels': [number_first_layer_channels] + params['perception']['conv']['channels'],
                    'kernels': params['perception']['conv']['kernels'],
                    'strides': params['perception']['conv']['strides'],
                    'dropouts': params['perception']['conv']['dropouts'],
                    'end_layer': True
                }
            )

            perception_fc = FC(
                params={
                    'neurons': [perception_convs.get_conv_output(sensor_input_shape)] +
                    params['perception']['fc']['neurons'],
                    'dropouts': params['perception']['fc']['dropouts'],
                    'end_layer': False
                }
            )

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params['perception']['fc']['neurons'][-1]

        elif 'res' in params['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('core.utils.model_utils.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])
            self.perception = resnet_module(
                pretrained=cfg.PRE_TRAINED, num_classes=params['perception']['res']['num_classes']
            )
            number_output_neurons = params['perception']['res']['num_classes']

        else:

            raise ValueError("invalid convolution layer type")

        self.measurements = FC(
            params={
                'neurons': [len(cfg.INPUTS)] + params['measurements']['fc']['neurons'],
                'dropouts': params['measurements']['fc']['dropouts'],
                'end_layer': False
            }
        )

        self.join = Join(
            params={
                'after_process': FC(
                    params={
                        'neurons': [params['measurements']['fc']['neurons'][-1] + number_output_neurons] +
                        params['join']['fc']['neurons'],
                        'dropouts': params['join']['fc']['dropouts'],
                        'end_layer': False
                    }
                ),
                'mode': 'cat'
            }
        )

        self.speed_branch = FC(
            params={
                'neurons': [params['join']['fc']['neurons'][-1]] + params['speed_branch']['fc']['neurons'] + [1],
                'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                'end_layer': True
            }
        )

        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(
                FC(
                    params={
                        'neurons': [params['join']['fc']['neurons'][-1]] + params['branches']['fc']['neurons'] +
                        [len(cfg.TARGETS)],
                        'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                        'end_layer': True
                    }
                )
            )

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically

        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, a):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x)
        # Not a variable, just to store intermediate layers for future vizualization
        # self.intermediate_layers = inter
        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        """ Join measurements and perception"""
        j = self.join(x, m)

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(x)

        # We concatenate speed with the rest.
        return branch_outputs + [speed_branch_output]

    def forward_branch(self, x, a, branch_number):
        """
        DO a forward operation and return a single branch.

        Args:
            x: the image input
            a: speed measurement
            branch_number: the branch number to be returned

        Returns:
            the forward operation on the selected branch

        """
        # Convert to integer just in case .
        # TODO: take four branches, this is hardcoded
        output_vec = torch.stack(self.forward(x, a)[0:4])

        return self.extract_branch(output_vec, branch_number)

    def get_perception_layers(self, x):
        return self.perception.get_layers_features(x)

    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number, torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]


class COILModel():

    def __init__(self, cfg):
        self._cfg = cfg
        self.checkpoint = torch.load(self._cfg.model.ckpt_path)

        # do the merge here

        self._model = CoILModule(self._cfg)
        self.first_iter = True
        # Load the model and prepare set it for evaluation
        self._model.load_state_dict(self.checkpoint['state_dict'])
        self._model.cuda()
        self._model.eval()

    def run_step(self, observations):

        sensor_data = observations['rgb'].float().cuda()
        directions = observations['command'] + 1.0
        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = observations['speed'] / 25.
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self._model.forward_branch(sensor_data, norm_speed, directions_tensor)

        # actions = self._process_model_outputs(model_outputs)

        return model_outputs

        # There is the posibility to replace some of the predictions with oracle predictions.

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        actions = []
        for output in outputs:

            steer, throttle, brake = output[0], output[1], output[2]
            if brake < 0.05:
                brake = 0.0

            if throttle > brake:
                brake = 0.0

            action = {'steer': float(steer), 'throttle': float(throttle), 'brake': float(brake)}
        actions.append(action)

        return actions


def CoILModule(cfg):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if cfg.MODEL_TYPE == 'coil-icra':

        return CoILICRA(cfg)

    else:

        raise ValueError(" Not found architecture name")
