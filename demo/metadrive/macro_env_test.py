import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from metadrive.constants import HELP_MESSAGE
from core.envs import MetaDriveMacroEnv


def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()


def main(args):
    config = dict(
        use_render=True,
    )
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = MetaDriveMacroEnv(config)

    obs = env.reset()
    print(HELP_MESSAGE)
    env.vehicle.expert_takeover = True
    for i in range(10):
        action = random.randint(0, 5)
        print("action is: {}".format(env.action_type.actions[action]))
        obs, rew, done, info = env.step(action)
        # draw_multi_channels_top_down_observation(obs, show_time=2)
        print('reward is: {}'.format(rew))
        if done or info["arrive_dest"]:
            env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="birdview", choices=["lidar", "rgb_camera", "birdview"])
    args = parser.parse_args()
    main(args)
