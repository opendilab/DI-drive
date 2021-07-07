from enum import Enum
import math
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

Orders = Enum("Order", "Follow_Lane Straight Right Left ChangelaneLeft ChangelaneRight")


def create_resnet_basic_block(width_output_feature_map, height_output_feature_map, nb_channel_in, nb_channel_out):
    basic_block = nn.Sequential(
        nn.Upsample(size=(width_output_feature_map, height_output_feature_map), mode="nearest"),
        nn.Conv2d(
            nb_channel_in,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            nb_channel_out,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )
    return basic_block


class ImplicitSupervisedModel(nn.Module):

    def __init__(
        self,
        nb_images_input,
        nb_images_output,
        hidden_size,
        nb_class_segmentation,
        nb_class_dist_to_tl,
        crop_sky=False,
    ):
        super().__init__()
        if crop_sky:
            self.size_state_RL = 6144
        else:
            self.size_state_RL = 8192
        resnet18 = models.resnet18(pretrained=False)

        # See https://arxiv.org/abs/1606.02147v1 section 4: Information-preserving
        # dimensionality changes
        #
        # "When downsampling, the first 1x1 projection of the convolutional branch is performed
        # with a stride of 2 in both dimensions, which effectively discards 75% of the input.
        # Increasing the filter size to 2x2 allows to take the full input into consideration,
        # and thus improves the information flow and accuracy."

        assert resnet18.layer2[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer3[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer4[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        resnet18.layer2[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer3[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer4[0].downsample[0].kernel_size = (2, 2)

        assert resnet18.layer2[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer3[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer4[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        new_conv1 = nn.Conv2d(nb_images_input * 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet18.conv1 = new_conv1

        self.encoder = torch.nn.Sequential(*(list(resnet18.children())[:-2]))  # resnet18_no_fc_no_avgpool
        self.last_conv_downsample = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.fc1_traffic_light_inters = nn.Linear(self.size_state_RL, hidden_size)
        self.fc2_tl_inters_none = nn.Linear(
            hidden_size, 4
        )  # Classification traffic_light US, traffic_light EU, intersection, none
        self.fc2_traffic_light_state = nn.Linear(
            hidden_size, 2
        )  # Classification red/orange or green (maybe 3 classes in stead of 2?)
        self.fc2_distance_to_tl = nn.Linear(
            hidden_size, nb_class_dist_to_tl
        )  # classification on the distance to traffic_light

        self.fc1_delta_y_yaw_camera = nn.Linear(
            self.size_state_RL, int(hidden_size / 4)
        )  # Hard coded here, we want a little hidden size...
        self.fc2_delta_y_yaw_camera = nn.Linear(
            int(hidden_size / 4), 2 * nb_images_output
        )  # Regression on delta_y and delta_yaw of each input frame!

        # We will upsample image with nearest neightboord interpolation between each umsample block
        # https://distill.pub/2016/deconv-checkerboard/
        self.up_sampled_block_0 = create_resnet_basic_block(8, 8, 512, 512)
        self.up_sampled_block_1 = create_resnet_basic_block(16, 16, 512, 256)
        self.up_sampled_block_2 = create_resnet_basic_block(32, 32, 256, 128)
        self.up_sampled_block_3 = create_resnet_basic_block(64, 64, 128, 64)
        self.up_sampled_block_4 = create_resnet_basic_block(128, 128, 64, 32)

        self.last_conv_segmentation = nn.Conv2d(
            32,
            nb_class_segmentation * nb_images_output,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
        )
        self.last_bn = nn.BatchNorm2d(
            nb_class_segmentation * nb_images_output,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, batch_image):
        # Encoder first, resnet18 without last fc and abg pooling
        encoding = self.encoder(batch_image)  # 512*4*4 or 512*4*3 (crop sky)

        encoding = self.last_conv_downsample(encoding)

        # Segmentation branch
        upsample0 = self.up_sampled_block_0(encoding)  # 512*8*8 or 512*6*8 (crop sky)
        upsample1 = self.up_sampled_block_1(upsample0)  # 256*16*16 or 256*12*16 (crop sky)
        upsample2 = self.up_sampled_block_2(upsample1)  # 128*32*32 or 128*24*32 (crop sky)
        upsample3 = self.up_sampled_block_3(upsample2)  # 64*64*64 or 64*48*64 (crop sky)
        upsample4 = self.up_sampled_block_4(upsample3)  # 32*128*128 or 32*74*128 (crop sky)

        out_seg = self.last_bn(self.last_conv_segmentation(upsample4))  # nb_class_segmentation*128*128

        # Classification branch, traffic_light (+ state), intersection or none
        classif_state_net = encoding.view(-1, self.size_state_RL)

        traffic_light_state_net = self.fc1_traffic_light_inters(classif_state_net)
        traffic_light_state_net = nn.functional.relu(traffic_light_state_net)

        classif_output = self.fc2_tl_inters_none(traffic_light_state_net)
        state_output = self.fc2_traffic_light_state(traffic_light_state_net)
        dist_to_tl_output = self.fc2_distance_to_tl(traffic_light_state_net)

        delta_position_yaw_state = self.fc1_delta_y_yaw_camera(classif_state_net)
        delta_position_yaw_state = nn.functional.relu(delta_position_yaw_state)
        delta_position_yaw_output = self.fc2_delta_y_yaw_camera(delta_position_yaw_state)

        return out_seg, classif_output, state_output, dist_to_tl_output, delta_position_yaw_output


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.cuda.FloatTensor(size).normal_()
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class ImplicitDQN(nn.Module):

    def __init__(
        self, action_space, history_length=4, quantile_embedding_dim=64, crop_sky=False, num_quantile_samples=32
    ):
        super().__init__()
        self.action_space = action_space
        self.history_length = history_length

        self.magic_number_repeat_scaler_in_fc = 10
        self.magic_number_SCALE_steering_in_fc = 10  # We want to multiply by 10 the steering...
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_samples = num_quantile_samples

        if crop_sky:
            size_RL_state = 6144
        else:
            size_RL_state = 8192
        self.iqn_fc = nn.Linear(self.quantile_embedding_dim, size_RL_state)

        hidden_size = 1024

        self.fcnoisy_h_a = NoisyLinear(size_RL_state, hidden_size)

        hidden_size2 = 512

        self.fcnoisy0_z_a_lane_follow = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length +
            4 * self.magic_number_repeat_scaler_in_fc,
            hidden_size2,
        )
        self.fcnoisy0_z_a_straight = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length +
            4 * self.magic_number_repeat_scaler_in_fc,
            hidden_size2,
        )
        self.fcnoisy0_z_a_right = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length +
            4 * self.magic_number_repeat_scaler_in_fc,
            hidden_size2,
        )
        self.fcnoisy0_z_a_left = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length +
            4 * self.magic_number_repeat_scaler_in_fc,
            hidden_size2,
        )
        self.fcnoisy0_z_a_lane_right = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length +
            4 * self.magic_number_repeat_scaler_in_fc,
            hidden_size2,
        )
        self.fcnoisy0_z_a_lane_left = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length +
            4 * self.magic_number_repeat_scaler_in_fc,
            hidden_size2,
        )

        self.fcnoisy1_z_a_lane_follow = NoisyLinear(hidden_size2, action_space)

        self.fcnoisy1_z_a_lane_follow = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_straight = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_right = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_left = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_lane_right = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_lane_left = NoisyLinear(hidden_size2, action_space)

    def forward(self, observations):
        if 'obs' in observations:
            observations = observations['obs']
        speeds = observations['speed'].cuda()
        steerings = observations['steer'].cuda()
        images = observations['image'].cuda()
        targets = observations['targets'].cuda()
        orders = observations['order']
        num_quantiles = self.num_quantile_samples

        batch_size = images.shape[0]

        quantiles = torch.cuda.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1)

        quantile_net = quantiles.repeat([1, self.quantile_embedding_dim])

        quantile_net = torch.cos(
            torch.arange(
                1,
                self.quantile_embedding_dim + 1,
                1,
                device=torch.device("cuda"),
                dtype=torch.float32,
            ) * math.pi * quantile_net
        )

        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = F.relu(quantile_net)

        rl_state_net = images.repeat(num_quantiles, 1)
        rl_state_net = rl_state_net * quantile_net

        mask_lane_follow = orders == Orders.Follow_Lane.value
        mask_straight = orders == Orders.Straight.value
        mask_right = orders == Orders.Right.value
        mask_left = orders == Orders.Left.value
        mask_lane_right = orders == Orders.ChangelaneRight.value
        mask_lane_left = orders == Orders.ChangelaneLeft.value

        if batch_size != 1:
            mask_lane_follow = mask_lane_follow.cuda()
            mask_straight = mask_straight.cuda()
            mask_right = mask_right.cuda()
            mask_left = mask_left.cuda()
            mask_lane_right = mask_lane_right.cuda()
            mask_lane_left = mask_lane_left.cuda()

            mask_lane_follow = mask_lane_follow.float()[:, None].repeat(num_quantiles, 1)
            mask_straight = mask_straight.float()[:, None].repeat(num_quantiles, 1)
            mask_right = mask_right.float()[:, None].repeat(num_quantiles, 1)
            mask_left = mask_left.float()[:, None].repeat(num_quantiles, 1)
            mask_lane_right = mask_lane_right.float()[:, None].repeat(num_quantiles, 1)
            mask_lane_left = mask_lane_left.float()[:, None].repeat(num_quantiles, 1)

        else:
            mask_lane_follow = bool(mask_lane_follow)
            mask_straight = bool(mask_straight)
            mask_right = bool(mask_right)
            mask_left = bool(mask_left)
            mask_lane_right = bool(mask_lane_right)
            mask_lane_left = bool(mask_lane_left)

        just_before_order_heads_a = F.relu(self.fcnoisy_h_a(rl_state_net))

        steerings = steerings * self.magic_number_SCALE_steering_in_fc

        speeds = speeds.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)
        steerings = steerings.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)
        targets = targets.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)

        just_before_order_heads_a_plus_speed_steering = torch.cat(
            (just_before_order_heads_a, speeds, steerings, targets), 1
        )

        a_lane_follow = self.fcnoisy0_z_a_lane_follow(just_before_order_heads_a_plus_speed_steering)
        a_lane_follow = self.fcnoisy1_z_a_lane_follow(F.relu(a_lane_follow))

        a_straight = self.fcnoisy0_z_a_straight(just_before_order_heads_a_plus_speed_steering)
        a_straight = self.fcnoisy1_z_a_straight(F.relu(a_straight))

        a_right = self.fcnoisy0_z_a_right(just_before_order_heads_a_plus_speed_steering)
        a_right = self.fcnoisy1_z_a_right(F.relu(a_right))

        a_left = self.fcnoisy0_z_a_left(just_before_order_heads_a_plus_speed_steering)
        a_left = self.fcnoisy1_z_a_left(F.relu(a_left))

        a_lane_right = self.fcnoisy0_z_a_lane_right(just_before_order_heads_a_plus_speed_steering)
        a_lane_right = self.fcnoisy1_z_a_lane_right(F.relu(a_lane_right))

        a_lane_left = self.fcnoisy0_z_a_lane_left(just_before_order_heads_a_plus_speed_steering)
        a_lane_left = self.fcnoisy1_z_a_lane_left(F.relu(a_lane_left))

        a = a_lane_follow * mask_lane_follow \
            + a_straight * mask_straight \
            + a_right * mask_right \
            + a_left * mask_left \
            + a_lane_right * mask_lane_right \
            + a_lane_left * mask_lane_left
        a = a.view(num_quantiles, batch_size, -1)
        a = a.mean(0)
        return {'logit': a}  #quantiles

    def reset_noise(self):
        for name, module in self.named_children():
            if "fcnoisy" in name:
                module.reset_noise()
