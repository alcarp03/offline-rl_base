# pylint: disable=unused-import,too-many-return-statements

import os
import random
import re
from typing import List, Tuple
from urllib import request
import pickle
import gym
import math
import numpy as np
from pathlib import Path
from .dataset import Episode, MDPDataset, Transition
from .envs import ChannelFirst
from PIL import Image

DATA_DIRECTORY = "d3rlpy_data"
DROPBOX_URL = "https://www.dropbox.com/s"
CARTPOLE_URL = f"{DROPBOX_URL}/uep0lzlhxpi79pd/cartpole_v1.1.0.h5?dl=1"
CARTPOLE_RANDOM_URL = f"{DROPBOX_URL}/4lgai7tgj84cbov/cartpole_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long
PENDULUM_URL = f"{DROPBOX_URL}/ukkucouzys0jkfs/pendulum_v1.1.0.h5?dl=1"
PENDULUM_RANDOM_URL = f"{DROPBOX_URL}/hhbq9i6ako24kzz/pendulum_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long


class MyWaypoint:
    def __init__(self, x, y, heading):
        self.pos = np.array([x, y])
        self.heading = heading

def gen_my_wp_from_obs(raw_obs):
    return MyWaypoint(raw_obs.ego_vehicle_state.position[0],
                      raw_obs.ego_vehicle_state.position[1],
                      float(raw_obs.ego_vehicle_state.heading))


def gen_my_wp_from_path(wp):
    return MyWaypoint(wp.pos[0],
                      wp.pos[1],
                      float(wp.heading))

def gen_wp_path(vehicle_data, step, gap_threshold=0.3):
    data = list(vehicle_data.values())[step:]  # 这里step应该从0开始计数
    lane_index = data[0].ego_vehicle_state.lane_index
    try:
        current_path = data[0].waypoint_paths[lane_index]
    except:
        current_path = []
    path_wps = [gen_my_wp_from_path(wp) for wp in current_path]
    future_distance_travelled = ([obs.distance_travelled for obs in data[1:]] + [0])
    future_distance_travelled = [obs.distance_travelled for obs in data[1:]]
    track_wps = [gen_my_wp_from_obs(data[0])]
    acc_distance_travelled = 0
    for i, raw_obs in enumerate(data[1:]):
        wp = gen_my_wp_from_obs(raw_obs)
        acc_distance_travelled += raw_obs.distance_travelled
        if acc_distance_travelled >= gap_threshold:
            track_wps.append(wp)
            acc_distance_travelled = 0
    if len(track_wps) >= 5 + 1:
        return track_wps[1:6]
    elif len(path_wps) >= 6 and cal_wps_bias(current_path) < 0.1:
        return path_wps[1:6]
    else:
        return None

def cal_wps_bias(current_path):
    headings = []
    if len(headings) < 5:
        return 1e5
    for wp in current_path[:5]:
        headings.append(wp.heading)

    abs_bias = 0
    for i, heading in enumerate(headings[:-1]):
        next_heading = headings[i + 1]
        abs_bias += abs(next_heading - heading)
    return abs_bias


def goal_region_reward(threshold, goal_x, goal_y, cur_x, cur_y):
    eucl_distance = math.sqrt((goal_x - cur_x) ** 2 + (goal_y - cur_y) ** 2)

    if eucl_distance <= threshold:
        return 10
    else:
        return 0


def inside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ratio = 256 / 50  # 256 pixels corresonds to 50 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        x_pixel_loc = min(
            128 + round(x_diff * ratio), 255
        )  # cap on 256 which is the right edge
        y_pixel_loc = max(
            127 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        x_pixel_loc = max(
            127 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = max(
            127 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        x_pixel_loc = max(
            127 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = min(
            128 + round(y_diff * ratio), 255
        )  # cap on 256 which is the bottom edge

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        x_pixel_loc = min(
            128 + round(x_diff * ratio), 255
        )  # cap on 256 which is the right edge
        y_pixel_loc = min(
            128 + round(y_diff * ratio), 255
        )  # cap on 256 which is the bottom edge

    # To find if goal is at cur (do not change to elif)
    if (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and (
        abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05
    ):
        x_pixel_loc = 128
        y_pixel_loc = 128

    # On x-axis
    elif (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = min(128 + round(x_diff * ratio), 255)
        else:
            x_pixel_loc = max(127 - round(x_diff * ratio), 0)
        y_pixel_loc = min(128 + round(y_diff * ratio), 255)

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = max(127 - round(y_diff * ratio), 0)
        else:
            y_pixel_loc = min(128 + round(y_diff * ratio), 255)
        x_pixel_loc = min(128 + round(x_diff * ratio), 255)

    goal_obs = np.zeros((1, 256, 256))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 255
    return goal_obs


def outside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ratio = 256 / 50  # 256 pixels corresonds to 25 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = max(127 - round((25 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(128 + round((25 / (y_diff / x_diff)) * ratio), 255)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = 0

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = max(127 - round((25 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(127 - round((25 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 0

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = min(128 + round((25 * (y_diff / x_diff)) * ratio), 255)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(127 - round((25 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 255
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 255

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = min(128 + round((25 * (y_diff / x_diff)) * ratio), 255)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(128 + round((25 / (y_diff / x_diff)) * ratio), 255)
            y_pixel_loc = 255
        elif theta == (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = 255

    # On x-axis (do not change to elif)
    if (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = 255
        else:
            x_pixel_loc = 0
        y_pixel_loc = 128

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = 0
        else:
            y_pixel_loc = 255
        x_pixel_loc = 128

    goal_obs = np.zeros((1, 256, 256))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 255
    return goal_obs


def get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading):

    if 0 < cur_heading < math.pi:  # Facing Left Half
        theta = cur_heading

    elif -(math.pi) < cur_heading < 0:  # Facing Right Half
        theta = 2 * math.pi + cur_heading

    elif cur_heading == 0:  # Facing up North
        theta = 0

    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):  # Facing South
        theta = 2 * math.pi + cur_heading

    trans_matrix = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    cur_pos = np.array([[cur_x], [cur_y]])
    goal_pos = np.array([[goal_x], [goal_y]])
    trans_cur = np.round(np.matmul(trans_matrix, cur_pos), 5)
    trans_goal = np.round(np.matmul(trans_matrix, goal_pos), 5)

    return [trans_cur, trans_goal]


# mark goal position with integer 256, other entries are all 0
def get_goal_layer(goal_x, goal_y, cur_x, cur_y, cur_heading):

    trans_coor = get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading)
    trans_cur = trans_coor[0]
    trans_goal = trans_coor[1]

    if (trans_cur[0, 0] - 25) <= trans_goal[0, 0] <= (trans_cur[0, 0] + 25):
        if (trans_cur[1, 0] - 25) <= trans_goal[1, 0] <= (trans_cur[1, 0] + 25):
            inside = True
        else:
            inside = False
    else:
        inside = False

    if inside:
        goal_obs = inside_coor_to_pixel(
            trans_goal[0, 0], trans_goal[1, 0], trans_cur[0, 0], trans_cur[1, 0]
        )
    else:
        goal_obs = outside_coor_to_pixel(
            trans_goal[0, 0], trans_goal[1, 0], trans_cur[0, 0], trans_cur[1, 0]
        )

    return goal_obs


def global_target_pose(action, agent_obs):

    cur_x = agent_obs["ego"]["pos"][0]
    cur_y = agent_obs["ego"]["pos"][1]
    cur_heading = agent_obs["ego"]["heading"]

    if 0 < cur_heading < math.pi:  # Facing Left Half
        theta = cur_heading

    elif -(math.pi) < cur_heading < 0:  # Facing Right Half
        theta = 2 * math.pi + cur_heading

    elif cur_heading == 0:  # Facing up North
        theta = 0

    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):  # Facing South
        theta = 2 * math.pi + cur_heading

    trans_matrix = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    action_bev = np.array([[action[0]], [action[1]]])
    action_global = np.matmul(np.transpose(trans_matrix), action_bev)
    target_pose = np.array(
        [
            cur_x + action_global[0],
            cur_y + action_global[1],
            action[2] + cur_heading,
            0.1,
        ],
        dtype=object,
    )

    return

def get_cartpole(dataset_type: str = "replay") -> Tuple[MDPDataset, gym.Env]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if dataset_type == "replay":
        url = CARTPOLE_URL
        file_name = "cartpole_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = CARTPOLE_RANDOM_URL
        file_name = "cartpole_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading cartpole.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    dataset = MDPDataset.load(data_path)

    # environment
    env = gym.make("CartPole-v0")

    return dataset, env


def get_pendulum(dataset_type: str = "replay") -> Tuple[MDPDataset, gym.Env]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if dataset_type == "replay":
        url = PENDULUM_URL
        file_name = "pendulum_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = PENDULUM_RANDOM_URL
        file_name = "pendulum_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading pendulum.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    dataset = MDPDataset.load(data_path)

    # environment
    env = gym.make("Pendulum-v0")

    return dataset, env


def get_atari(env_name: str) -> Tuple[MDPDataset, gym.Env]:
    """Returns atari dataset and envrironment.

    The dataset is provided through d4rl-atari. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_atari

        dataset, env = get_atari('breakout-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        env_name: environment id of d4rl-atari dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_atari  # type: ignore

        env = ChannelFirst(gym.make(env_name))
        dataset = MDPDataset(discrete_action=True, **env.get_dataset())
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_atari_transitions(
    game_name: str, fraction: float = 0.01, index: int = 0
) -> Tuple[List[Transition], gym.Env]:
    """Returns atari dataset as a list of Transition objects and envrironment.

    The dataset is provided through d4rl-atari.
    The difference from ``get_atari`` function is that this function will
    sample transitions from all epochs.
    This function is necessary for reproducing Atari experiments.

    .. code-block:: python

        from d3rlpy.datasets import get_atari_transitions

        # get 1% of transitions from all epochs (1M x 50 epoch x 1% = 0.5M)
        dataset, env = get_atari_transitions('breakout', fraction=0.01)

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        game_name: Atari 2600 game name in lower_snake_case.
        fraction: fraction of sampled transitions.
        index: index to specify which trial to load.

    Returns:
        tuple of a list of :class:`d3rlpy.dataset.Transition` and gym
        environment.

    """
    try:
        import d4rl_atari

        # each epoch consists of 1M steps
        num_transitions_per_epoch = int(1000000 * fraction)

        transitions = []
        for i in range(50):
            env = gym.make(
                f"{game_name}-epoch-{i + 1}-v{index}", sticky_action=True
            )
            dataset = MDPDataset(discrete_action=True, **env.get_dataset())
            episodes = list(dataset.episodes)

            # copy episode data to release memory of unused data
            random.shuffle(episodes)
            num_data = 0
            copied_episodes = []
            for episode in episodes:
                copied_episode = Episode(
                    observation_shape=tuple(episode.get_observation_shape()),
                    action_size=episode.get_action_size(),
                    observations=episode.observations.copy(),
                    actions=episode.actions.copy(),
                    rewards=episode.rewards.copy(),
                    terminal=episode.terminal,
                )
                copied_episodes.append(copied_episode)

                num_data += len(copied_episode)
                if num_data > num_transitions_per_epoch:
                    break

            transitions_per_epoch = []
            for episode in copied_episodes:
                transitions_per_epoch += episode.transitions
            transitions += transitions_per_epoch[:num_transitions_per_epoch]

        return transitions, ChannelFirst(env)
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_d4rl(env_name: str) -> Tuple[MDPDataset, gym.Env]:
    """Returns d4rl dataset and envrironment.

    The dataset is provided through d4rl.

    .. code-block:: python

        from d3rlpy.datasets import get_d4rl

        dataset, env = get_d4rl('hopper-medium-v0')

    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl

    Args:
        env_name: environment id of d4rl dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        dataset = env.get_dataset()

        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        timeouts = dataset["timeouts"]
        episode_terminals = np.logical_or(terminals, timeouts)

        mdp_dataset = MDPDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
        )

        return mdp_dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n"
            "pip install git+https://github.com/rail-berkeley/d4rl"
        ) from e


ATARI_GAMES = [
    "adventure",
    "air-raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank-heist",
    "battle-zone",
    "beam-rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper-command",
    "crazy-climber",
    "defender",
    "demon-attack",
    "double-dunk",
    "elevator-action",
    "enduro",
    "fishing-derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice-hockey",
    "jamesbond",
    "journey-escape",
    "kangaroo",
    "krull",
    "kung-fu-master",
    "montezuma-revenge",
    "ms-pacman",
    "name-this-game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private-eye",
    "qbert",
    "riverraid",
    "road-runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space-invaders",
    "star-gunner",
    "tennis",
    "time-pilot",
    "tutankham",
    "up-n-down",
    "venture",
    "video-pinball",
    "wizard-of-wor",
    "yars-revenge",
    "zaxxon",
]

def get_waymo(env_name, type, input_path: str) -> MDPDataset:
    try:
        if type == 'pixel':



            n_scenarios = 2  # number of training scenarios
            n_vehicles= 2  # number of vehicles to examine

            scenarios = list()
            for scenario_name in os.listdir(input_path):
                scenarios.append(scenario_name)

            save_directory = Path(__file__).absolute().parents[0] / "d3rlpy_logs"

            if not os.path.isdir(save_directory):
                index = 0
                os.mkdir(save_directory)
            else:
                index = len(os.listdir(save_directory))

            if n_scenarios == "max" or n_scenarios > len(scenarios):
                n_scenarios = len(scenarios)

            for scenario in scenarios[index:n_scenarios]:
                obs = list()
                actions = list()
                rewards = list()
                terminals = list()
                print(f"Processing scenario {scenario}.")
                vehicle_ids = list()

                scenario_path = Path(input_path) / scenario
                for filename in os.listdir(scenario_path):
                    if filename.endswith(".pkl"):
                        match = re.search("(.*).pkl", filename)
                        assert match is not None
                        vehicle_id = match.group(1)
                        if vehicle_id not in vehicle_ids:
                            vehicle_ids.append(vehicle_id)

                if len(vehicle_ids) < 2:
                    continue

                if n_vehicles == "max" or n_vehicles > len(vehicle_ids):
                    n_vehicles = len(vehicle_ids)

                for id in vehicle_ids[0:n_vehicles]:
                    print(f"Adding data for vehicle id {id} in scenario {scenario}.")

                    with open(scenario_path / f"{id}.pkl", "rb") as f:
                        vehicle_data = pickle.load(f)

                    image_names = list()
                    for filename in os.listdir(scenario_path):
                        if filename.endswith(f"{id}.png"):
                            image_names.append(filename)
                    image_names = sorted(image_names)

                    goal_pos_x, goal_pos_y = vehicle_data[float(
                        image_names[0].split("_")[0])].ego_vehicle_state.mission.goal.position.as_np_array[0:2]
                    threshold = 3

                    for i in range(len(image_names) - 1):
                        with Image.open(scenario_path / image_names[i], "r") as image:
                            image.seek(0)
                            sim_time = image_names[i].split("_")[0]
                            sim_time_next = image_names[i + 1].split("_")[0]
                            current_position = vehicle_data[float(sim_time)].ego_vehicle_state.position
                            current_heading = vehicle_data[float(sim_time)].ego_vehicle_state.heading
                            next_position = vehicle_data[float(sim_time_next)].ego_vehicle_state.position
                            next_heading = vehicle_data[float(sim_time_next)].ego_vehicle_state.heading
                            trans_coor = get_trans_coor(
                                next_position[0],
                                next_position[1],
                                current_position[0],
                                current_position[1],
                                current_heading,
                            )
                            trans_cur = trans_coor[0]
                            trans_next = trans_coor[1]
                            dx = trans_next[0, 0] - trans_cur[0, 0]
                            dy = trans_next[1, 0] - trans_cur[1, 0]
                            dheading = next_heading - current_heading
                            events = vehicle_data[float(sim_time)].events
                            terminal = 0
                            for name, value in events._asdict().items():
                                if ((name == "collisions") or
                                    (name == "off_road") or
                                    (name == "reached_goal") or
                                    (name == "reached_max_episode_steps")
                                ) and bool(value):
                                    terminal = 1
                                    break

                            bev = np.moveaxis(np.asarray(image), -1, 0)
                            goal_obs = get_goal_layer(
                                goal_pos_x,
                                goal_pos_y,
                                current_position[0],
                                current_position[1],
                                current_heading,
                            )
                            extended_ob = np.concatenate((bev, goal_obs), axis=0)
                            obs.append(extended_ob)
                            actions.append([dx, dy, dheading])
                            dist_reward = vehicle_data[float(sim_time)].distance_travelled
                            goal_reward = goal_region_reward(
                                threshold,
                                goal_pos_x,
                                goal_pos_y,
                                current_position[0],
                                current_position[1],
                            )
                            rewards.append(dist_reward + goal_reward)

                            terminals.append(terminal)

                    print(str(len(obs)) + " pieces of data are added into dataset.")
                    # n_vehicles = train_config[
                    #     "n_vehicles"
                    # ]  # Return to default value for next scenario processing

                obs = np.array(obs, dtype=np.uint8)
                actions = np.array(actions)
                rewards = np.array(rewards)
                terminals = np.array(terminals)
                episode_terminals = terminals.copy()
                episode_terminals[-1] = 1
                dataset = MDPDataset(obs, actions, rewards, terminals, episode_terminals=episode_terminals)
                return dataset
        elif type == 'vector':
            # 读取SMARTS格式数据集，并将其转化为MDP Dataset
            n_scenarios = 2  # number of training scenarios
            n_vehicles = 2  # number of vehicles to examine
            scenarios = list()
            for scenario_name in os.listdir(input_path):
                scenarios.append(scenario_name)
            save_directory = Path(__file__).absolute().parents[0] / "d3rlpy_logs_{}".format(type)
            if not os.path.isdir(save_directory):
                index = 0
                os.mkdir(save_directory)
            else:
                index = len(os.listdir(save_directory))
            # on shoulder test
            # scenarios = ['a81d659e56d1d7b5']
            if n_scenarios == "max" or n_scenarios > len(scenarios):
                n_scenarios = len(scenarios)  # 2
            wash_data = True
            cnt = 0
            obs = list()
            actions = list()
            rewards = list()
            terminals = list()
            episode_terminals = list()
            turning_files = []
            straight_files = []
            for sc_index, scenario in enumerate(scenarios[0:n_scenarios]):
                if sc_index % 10 == 0:
                    print(f"Processing scenario {scenario}.")
                vehicle_ids = list()

                scenario_path = Path(input_path) / scenario
                for filename in os.listdir(scenario_path):
                    if filename.endswith(".pkl"):
                        match = re.search("vehicle-(.*).pkl", filename)
                        assert match is not None
                        vehicle_id = match.group(1)
                        if vehicle_id not in vehicle_ids:
                            vehicle_ids.append(vehicle_id)

                n_vehicles = len(vehicle_ids)
                for id in vehicle_ids[0:n_vehicles]:
                    if sc_index % 20 == 0:
                        print(f"Adding data for vehicle id {id} in scenario {scenario}.")
                    with open(
                        scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                        "rb",
                    ) as f:
                        vehicle_data = pickle.load(f)
                    image_names = list()

                    for filename in os.listdir(scenario_path):
                        if filename.endswith(f"-{id}.png"):
                            image_names.append(filename)

                    image_names = sorted(image_names)

                    goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[0]
                    goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[1]

                    threshold = 3
                    if wash_data:
                        dheadings = []
                        raw_headings = []

                        raw_headings = [obs.ego_vehicle_state.heading for obs in vehicle_data.values()]
                        all_ts = [float(name.split("_Agent")[0]) for name in image_names]
                        vehicle_ts = [obs.elapsed_sim_time for obs in vehicle_data.values()]
                        assert len(all_ts) == len(vehicle_data) and np.all(
                            np.array(vehicle_ts) == np.array(all_ts)), 'wrong ts'
                        ts, error_ts, correct_headings = [], [], []

                        for i, heading in enumerate(raw_headings):
                            is_outlier = True
                            neighbor_headings = []
                            if i > 0:
                                neighbor_headings += raw_headings[max(0, i - 3):i]
                            if i < len(raw_headings) - 1:
                                neighbor_headings += raw_headings[i + 1:i + 4]
                            for neighbor_heading in neighbor_headings:
                                if abs(heading - neighbor_heading) < 8 * math.pi / 180:  # 我们认为正常情况下不会有车辆在0.1s转弯8度
                                    is_outlier = False
                                    break
                            if is_outlier:
                                # error_ts.append(image_names)
                                error_ts.append((i + 1) / 10)  # 我们暂时假设，车辆的ts从0.1开始，以step=0.1不间断线性增长
                            else:
                                ts.append((i + 1) / 10)
                                correct_headings.append(heading)
                        fixed_headings = np.interp(error_ts, ts, correct_headings)  # 线性插值，修复离群点
                        for i, x in enumerate(error_ts):
                            ts.append(x)
                            correct_headings.append(fixed_headings[i])
                        ts_heading_pairs = zip(ts, correct_headings)
                        ts_heading_pairs = sorted(ts_heading_pairs, key=lambda outlier: outlier[0])
                        last_heading = ts_heading_pairs[0][1]
                        abs_dheadings = []
                        for t, heading in ts_heading_pairs[1:]:
                            abs_dheadings.append(
                                min(abs(heading - last_heading),
                                    2 * math.pi - abs(heading - last_heading)) * 180 / math.pi)
                            last_heading = heading


                        if len(raw_headings) < 3:
                            continue



                        if max(abs_dheadings) >= 0.5 and scenario not in turning_files:
                            turning_files.append(scenario)

            last_len = 0
            print('start processing turning/changing lane data')
            for sc_index, scenario in enumerate(turning_files):  # 完成清洗，开始抽取数据，构造MDPdataset
                if sc_index % 20 == 0:
                    print(f"Processing turning/changing lane scenario {scenario}.")
                vehicle_ids = list()

                scenario_path = Path(input_path) / scenario
                for filename in os.listdir(scenario_path):
                    if filename.endswith(".pkl"):
                        match = re.search("vehicle-(.*).pkl", filename)
                        assert match is not None
                        vehicle_id = match.group(1)
                        if vehicle_id not in vehicle_ids:
                            vehicle_ids.append(vehicle_id)

                n_vehicles = len(vehicle_ids)
                for id in vehicle_ids[0:n_vehicles]:
                    with open(
                        scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                        "rb",
                    ) as f:
                        vehicle_data = pickle.load(f)
                    image_names = list()

                    for filename in os.listdir(scenario_path):
                        if filename.endswith(f"-{id}.png"):
                            image_names.append(filename)

                    image_names = sorted(image_names)

                    goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[0]
                    goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[1]

                    threshold = 3

                    # 筛选出转弯/换道场景中位于转弯前后的数据
                    raw_headings = [obs.ego_vehicle_state.heading for obs in vehicle_data.values()]
                    all_ts = [float(name.split("_Agent")[0]) for name in image_names]
                    vehicle_ts = [obs.elapsed_sim_time for obs in vehicle_data.values()]
                    assert len(all_ts) == len(vehicle_data) and np.all(
                        np.array(vehicle_ts) == np.array(all_ts)), 'wrong ts'
                    ts, error_ts, correct_headings = [], [], []

                    for i, heading in enumerate(raw_headings):
                        is_outlier = True
                        neighbor_headings = []
                        if i > 0:
                            neighbor_headings += raw_headings[max(0, i - 3):i]
                        if i < len(raw_headings) - 1:
                            neighbor_headings += raw_headings[i + 1:i + 4]
                        for neighbor_heading in neighbor_headings:
                            if abs(heading - neighbor_heading) < 8 * math.pi / 180:  # 我们认为正常情况下不会有车辆在0.1s转弯8度
                                is_outlier = False
                                break
                        if is_outlier:
                            # error_ts.append(image_names)
                            error_ts.append((i + 1) / 10)  # 我们暂时假设，车辆的ts从0.1开始，以step=0.1不间断线性增长
                        else:
                            ts.append((i + 1) / 10)
                            correct_headings.append(heading)
                    fixed_headings = np.interp(error_ts, ts, correct_headings)  # 线性插值，修复离群点
                    for i, x in enumerate(error_ts):
                        ts.append(x)
                        correct_headings.append(fixed_headings[i])
                    ts_heading_pairs = zip(ts, correct_headings)
                    ts_heading_pairs = sorted(ts_heading_pairs, key=lambda outlier: outlier[0])
                    last_heading = ts_heading_pairs[0][1]
                    abs_dheadings = []
                    for t, heading in ts_heading_pairs[1:]:
                        abs_dheadings.append(
                            min(abs(heading - last_heading), 2 * math.pi - abs(heading - last_heading)) * 180 / math.pi)
                        last_heading = heading
                    abs_dheadings = ['None'] + abs_dheadings  # 加入占位符‘None’

                    for i in range(0, len(image_names) - 1):  # i的含义是step cnt，从0开始
                        with Image.open(scenario_path / image_names[i], "r") as image:
                            image.seek(0)
                            sim_time = image_names[i].split("_Agent")[0]
                            sim_time_next = image_names[i + 1].split("_Agent")[0]
                        if max(abs_dheadings[max(1, i - 10):i + 10 + 1]) < 0.5:  # 仅使用转弯/并道前后10个step的action
                            continue

                        # todo 确定observation，包括waypoint_obs和neighbor_obs
                        fixed_waypoints = gen_wp_path(vehicle_data, step=i, gap_threshold=0.3)
                        if fixed_waypoints is None:
                            continue  # 如果无法取得5个有效的wp，则抛弃当前step的数据
                        # todo 在这算两种obs，记得做坐标系转换
                        # ego_vehicle_state
                        current_raw_obs = vehicle_data[float(sim_time)]
                        state = current_raw_obs.ego_vehicle_state
                        pos = state.position[:2]
                        heading = float(state.heading)
                        speed = state.speed
                        lane_index = state.lane_index
                        rotate_M = np.array([
                            [np.cos(heading), np.sin(heading)],
                            [-np.sin(heading), np.cos(heading)]]
                        )  # heading是与y正半轴的夹角，这样转化是否会出现问题？

                        ego_lane_positions = np.array([wp.pos for wp in fixed_waypoints])
                        ego_lane_headings = np.array([float(wp.heading) for wp in fixed_waypoints])

                        all_lane_rel_position = (
                            (ego_lane_positions.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T).reshape(5, 2)
                        all_lane_rel_heading = (ego_lane_headings - heading)
                        all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi  # 这是在干嘛
                        all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

                        EnvInfo_rel_pos_heading = np.zeros((1, 15))
                        EnvInfo_speed_limit = np.zeros((1, 1))
                        EnvInfo_bounding_box = np.zeros((1, 2))
                        EnvInfo_lane_width = np.zeros((1, 1))
                        EnvInfo_rel_pos_heading[0, :10] = all_lane_rel_position.reshape(
                            10, )  # todo 看一下测试环境的wp[0].heading是否始终是0
                        EnvInfo_rel_pos_heading[0, 10:] = all_lane_rel_heading.reshape(5, )
                        try:
                            speed_limit = vehicle_data[float(sim_time)].waypoint_paths[lane_index][0].speed_limit
                            # 有时waymo数据集的lane_index会out of range
                        except:
                            speed_limit = vehicle_data[float(sim_time)].waypoint_paths[0][0].speed_limit
                        EnvInfo_speed_limit[0, 0] = speed_limit
                        EnvInfo_bounding_box[0, 0] = state.bounding_box.length
                        EnvInfo_bounding_box[0, 1] = state.bounding_box.width

                        EnvInfo = np.concatenate([
                            EnvInfo_rel_pos_heading,  # 15
                            EnvInfo_speed_limit,  # 1
                            EnvInfo_bounding_box,  # 2
                        ], -1).astype(np.float32)

                        # Neighbor Info
                        on_road_neighbors = []
                        for neighbor in current_raw_obs.neighborhood_vehicle_states:
                            if neighbor.lane_id != 'off_lane':
                                on_road_neighbors.append(neighbor)
                        on_road_neighbors = on_road_neighbors[:50]
                        neighbors_pos = np.zeros([50, 2])
                        neighbors_bounding_box = np.zeros([50, 2])
                        neighbors_speed = np.zeros(50)
                        neighbors_heading = np.zeros([50])
                        for nb_ind, neighbor in enumerate(on_road_neighbors):
                            neighbors_pos[nb_ind] = neighbor.position[:2]
                            neighbors_speed[nb_ind] = neighbor.speed
                            neighbors_heading[nb_ind] = neighbor.heading
                            neighbors_bounding_box[nb_ind, 0] = neighbor.bounding_box.length
                            neighbors_bounding_box[nb_ind, 1] = neighbor.bounding_box.width
                        nb_num = len(on_road_neighbors)
                        neighbors_rel_vel = np.zeros((50, 2))
                        neighbors_rel_vel[:nb_num, 0] = -np.sin(neighbors_heading[:nb_num]) * neighbors_speed[
                                                                                              :nb_num] + np.sin(
                            heading) * speed
                        neighbors_rel_vel[:nb_num, 1] = np.cos(neighbors_heading[:nb_num]) * neighbors_speed[
                                                                                             :nb_num] - np.cos(
                            heading) * speed

                        nb_mask = np.all(neighbors_pos == 0, -1).astype(np.float32)

                        neighbors_dist = np.sqrt(((neighbors_pos - pos) ** 2).sum(-1)) + nb_mask * 1e5
                        st = np.argsort(neighbors_dist)[:5]

                        NeighborInfo_rel_pos = ((neighbors_pos[st] - pos) @ rotate_M.T)
                        NeighborInfo_rel_vel = ((neighbors_rel_vel[st]) @ rotate_M.T)
                        NeighborInfo_rel_heading = (neighbors_heading - heading)[st].reshape(5, 1)
                        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
                        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
                        NeighborInfo_boundingbox = neighbors_bounding_box[st]

                        NeighborInfo = np.concatenate([
                            NeighborInfo_rel_pos,  # 2
                            NeighborInfo_rel_vel,  # 2
                            NeighborInfo_rel_heading,  # 1
                            NeighborInfo_boundingbox,  # 2
                        ], -1).astype(np.float32)

                        if len(on_road_neighbors) < 5:  # padding
                            NeighborInfo[len(on_road_neighbors):] = 0

                        obs.append(np.concatenate([
                            NeighborInfo.reshape(-1, ),  # (25)
                            EnvInfo.reshape(-1, ),  # (15)
                        ]))

                        # 确定action：相对速度小于0.2->0，不小于0.2->1
                        speed, heading, lane_index, lane_id = state.speed, state.heading, state.lane_index, state.lane_id
                        try:
                            speed_limit = current_raw_obs.waypoint_paths[lane_index][0].speed_limit
                            # 有时waymo数据集的lane_index会out of range
                        except:
                            speed_limit = current_raw_obs.waypoint_paths[0][0].speed_limit
                        if speed / speed_limit >= 0.2:
                            action = 1  # moving
                        else:
                            action = 0  # keeping still
                        actions.append(action)
                        # 确定terminal
                        next_raw_obs = vehicle_data[float(sim_time_next)]
                        events = next_raw_obs.events
                        # 使用next_obs判定是否到达目的地，若到达则terminal
                        next_position = vehicle_data[float(sim_time_next)].ego_vehicle_state.position
                        goal_reward = goal_region_reward(
                            threshold,
                            goal_pos_x,
                            goal_pos_y,
                            next_position[0],
                            next_position[1],
                        )  # 若到达终点则奖励+10，不确定是否要使用这一项
                        if float(sim_time) > 8.7:
                            aaa = 1
                        is_reached_goal = True if goal_reward > 0 else False
                        if (not (events.off_road or events.reached_goal or events.reached_max_episode_steps or
                                 is_reached_goal)) and len(events.collisions) == 0:
                            terminal = 0
                        else:
                            terminal = 1
                        terminals.append(terminal)
                        # 确定reward：[distance_travelled, action_reward, goal_reward]
                        dist_reward = next_raw_obs.distance_travelled
                        action_reward = {
                            0: 0,
                            1: 1
                        }[action]
                        reward = [dist_reward, action_reward, goal_reward]

                        rewards.append(reward)
                        episode_terminals.append(0)
                        if terminal == 1:
                            break
                    episode_terminals[-1] = 1  # 在每个episode的最后一个step，将episode_terminals的最后一个元素修改为1
                    print(f'ind {sc_index} scen {scenario} total_len {len(obs) - last_len} ter_sum {sum(episode_terminals)}')
                    last_len = len(obs)
            aaa = 1

            # 计算直行轨迹的数量和长度
            turning_trajectory_num = len(turning_files)
            turning_trajectory_len = len(obs) // turning_trajectory_num  # 转弯/换道数据集中轨迹平均长度
            all_straight_files = []
            for sc_index, scenario in enumerate(scenarios[0:n_scenarios]):
                if scenario not in turning_files:
                    all_straight_files.append(scenario)
            straight_trajectory_num = min(len(all_straight_files), turning_trajectory_num // 2)
            straight_trajectory_len = turning_trajectory_len

            seed = 20
            np.random.seed(seed)
            random.seed(seed)
            straight_idxs = np.random.choice(a=len(all_straight_files), size=straight_trajectory_num, replace=False,
                                             p=None).tolist()
            straight_files = [all_straight_files[idx] for idx in straight_idxs]  # 抽取直行scenario

            print('start processing straight data')
            last_len = len(obs)
            for sc_index, scenario in enumerate(straight_files):  # 开始抽取直行数据，构造MDPdataset
                if sc_index % 20 == 0:
                    print(f"Processing straight scenario {scenario}.")
                vehicle_ids = list()

                scenario_path = Path(input_path) / scenario
                for filename in os.listdir(scenario_path):
                    if filename.endswith(".pkl"):
                        match = re.search("vehicle-(.*).pkl", filename)
                        assert match is not None
                        vehicle_id = match.group(1)
                        if vehicle_id not in vehicle_ids:
                            vehicle_ids.append(vehicle_id)

                n_vehicles = len(vehicle_ids)
                for id in vehicle_ids[0:n_vehicles]:
                    with open(
                        scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                        "rb",
                    ) as f:
                        vehicle_data = pickle.load(f)
                    image_names = list()

                    for filename in os.listdir(scenario_path):
                        if filename.endswith(f"-{id}.png"):
                            image_names.append(filename)

                    image_names = sorted(image_names)

                    goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[0]
                    goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])].ego_vehicle_state.position[1]

                    threshold = 3

                    tmp_trajectory_len = min(len(vehicle_data) - 10, straight_trajectory_len)
                    if tmp_trajectory_len < 10:  # 抛弃过短的直行轨迹
                        continue

                    # 修复轨迹中异常的heading值
                    raw_headings = [obs.ego_vehicle_state.heading for obs in vehicle_data.values()]
                    all_ts = [float(name.split("_Agent")[0]) for name in image_names]
                    vehicle_ts = [obs.elapsed_sim_time for obs in vehicle_data.values()]
                    assert len(all_ts) == len(vehicle_data) and np.all(
                        np.array(vehicle_ts) == np.array(all_ts)), 'wrong ts'
                    ts, error_ts, correct_headings = [], [], []

                    for i, heading in enumerate(raw_headings):
                        is_outlier = True
                        neighbor_headings = []
                        if i > 0:
                            neighbor_headings += raw_headings[max(0, i - 3):i]
                        if i < len(raw_headings) - 1:
                            neighbor_headings += raw_headings[i + 1:i + 4]
                        for neighbor_heading in neighbor_headings:
                            if abs(heading - neighbor_heading) < 8 * math.pi / 180:  # 我们认为正常情况下不会有车辆在0.1s转弯8度
                                is_outlier = False
                                break
                        if is_outlier:
                            # error_ts.append(image_names)
                            error_ts.append((i + 1) / 10)  # 我们暂时假设，车辆的ts从0.1开始，以step=0.1不间断线性增长
                        else:
                            ts.append((i + 1) / 10)
                            correct_headings.append(heading)
                    fixed_headings = np.interp(error_ts, ts, correct_headings)  # 线性插值，修复离群点
                    for i, x in enumerate(error_ts):
                        ts.append(x)
                        correct_headings.append(fixed_headings[i])
                    ts_heading_pairs = zip(ts, correct_headings)
                    ts_heading_pairs = sorted(ts_heading_pairs, key=lambda outlier: outlier[0])
                    last_heading = ts_heading_pairs[0][1]
                    abs_dheadings = []
                    for t, heading in ts_heading_pairs[1:]:
                        abs_dheadings.append(
                            min(abs(heading - last_heading), 2 * math.pi - abs(heading - last_heading)) * 180 / math.pi)
                        last_heading = heading
                    abs_dheadings = ['None'] + abs_dheadings  # 加入占位符‘None’

                    trajectory_offset = random.randint(0, len(vehicle_data) - 10 - tmp_trajectory_len)
                    # 轨迹第一个step的index;尽量不使用最后10个step的数据，因为可能没有足够的future step来拟合waypoint

                    for i in range(0, len(image_names) - 1):  # i的含义是step cnt，从0开始
                        with Image.open(scenario_path / image_names[i], "r") as image:
                            image.seek(0)
                            sim_time = image_names[i].split("_Agent")[0]
                            sim_time_next = image_names[i + 1].split("_Agent")[0]
                        if i < trajectory_offset or i >= trajectory_offset + tmp_trajectory_len:  # 抽取tmp_trajectory_len个连续step的数据
                            continue

                        # todo 确定observation，包括waypoint_obs和neighbor_obs
                        fixed_waypoints = gen_wp_path(vehicle_data, step=i, gap_threshold=0.3)
                        if fixed_waypoints is None:
                            continue  # 如果无法取得5个有效的wp，则抛弃当前step的数据
                        # todo 在这算两种obs，记得做坐标系转换
                        # ego_vehicle_state
                        current_raw_obs = vehicle_data[float(sim_time)]
                        state = current_raw_obs.ego_vehicle_state
                        pos = state.position[:2]
                        heading = float(state.heading)
                        speed = state.speed
                        lane_index = state.lane_index
                        rotate_M = np.array([
                            [np.cos(heading), np.sin(heading)],
                            [-np.sin(heading), np.cos(heading)]]
                        )  # heading是与y正半轴的夹角，这样转化是否会出现问题？

                        ego_lane_positions = np.array([wp.pos for wp in fixed_waypoints])
                        ego_lane_headings = np.array([float(wp.heading) for wp in fixed_waypoints])

                        all_lane_rel_position = (
                            (ego_lane_positions.reshape(-1, 2) - pos.reshape(1, 2)) @ rotate_M.T).reshape(5, 2)
                        all_lane_rel_heading = (ego_lane_headings - heading)
                        all_lane_rel_heading[np.where(all_lane_rel_heading > np.pi)] -= np.pi  # 这是在干嘛
                        all_lane_rel_heading[np.where(all_lane_rel_heading < -np.pi)] += np.pi

                        EnvInfo_rel_pos_heading = np.zeros((1, 15))
                        EnvInfo_speed_limit = np.zeros((1, 1))
                        EnvInfo_bounding_box = np.zeros((1, 2))
                        EnvInfo_lane_width = np.zeros((1, 1))
                        EnvInfo_rel_pos_heading[0, :10] = all_lane_rel_position.reshape(
                            10, )  # todo 看一下测试环境的wp[0].heading是否始终是0
                        EnvInfo_rel_pos_heading[0, 10:] = all_lane_rel_heading.reshape(5, )
                        try:
                            speed_limit = vehicle_data[float(sim_time)].waypoint_paths[lane_index][0].speed_limit
                            # 有时waymo数据集的lane_index会out of range
                        except:
                            speed_limit = vehicle_data[float(sim_time)].waypoint_paths[0][0].speed_limit
                        EnvInfo_speed_limit[0, 0] = speed_limit
                        EnvInfo_bounding_box[0, 0] = state.bounding_box.length
                        EnvInfo_bounding_box[0, 1] = state.bounding_box.width

                        EnvInfo = np.concatenate([
                            EnvInfo_rel_pos_heading,  # 15
                            EnvInfo_speed_limit,  # 1
                            EnvInfo_bounding_box,  # 2
                        ], -1).astype(np.float32)

                        # Neighbor Info
                        on_road_neighbors = []
                        for neighbor in current_raw_obs.neighborhood_vehicle_states:
                            if neighbor.lane_id != 'off_lane':
                                on_road_neighbors.append(neighbor)
                        on_road_neighbors = on_road_neighbors[:50]
                        neighbors_pos = np.zeros([50, 2])
                        neighbors_bounding_box = np.zeros([50, 2])
                        neighbors_speed = np.zeros(50)
                        neighbors_heading = np.zeros([50])
                        for nb_ind, neighbor in enumerate(on_road_neighbors):
                            neighbors_pos[nb_ind] = neighbor.position[:2]
                            neighbors_speed[nb_ind] = neighbor.speed
                            neighbors_heading[nb_ind] = neighbor.heading
                            neighbors_bounding_box[nb_ind, 0] = neighbor.bounding_box.length
                            neighbors_bounding_box[nb_ind, 1] = neighbor.bounding_box.width
                        nb_num = len(on_road_neighbors)
                        neighbors_rel_vel = np.zeros((50, 2))
                        neighbors_rel_vel[:nb_num, 0] = -np.sin(neighbors_heading[:nb_num]) * neighbors_speed[
                                                                                              :nb_num] + np.sin(
                            heading) * speed
                        neighbors_rel_vel[:nb_num, 1] = np.cos(neighbors_heading[:nb_num]) * neighbors_speed[
                                                                                             :nb_num] - np.cos(
                            heading) * speed

                        nb_mask = np.all(neighbors_pos == 0, -1).astype(np.float32)

                        neighbors_dist = np.sqrt(((neighbors_pos - pos) ** 2).sum(-1)) + nb_mask * 1e5
                        st = np.argsort(neighbors_dist)[:5]

                        NeighborInfo_rel_pos = ((neighbors_pos[st] - pos) @ rotate_M.T)
                        NeighborInfo_rel_vel = ((neighbors_rel_vel[st]) @ rotate_M.T)
                        NeighborInfo_rel_heading = (neighbors_heading - heading)[st].reshape(5, 1)
                        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading > np.pi)] -= np.pi
                        NeighborInfo_rel_heading[np.where(NeighborInfo_rel_heading < -np.pi)] += np.pi
                        NeighborInfo_boundingbox = neighbors_bounding_box[st]

                        NeighborInfo = np.concatenate([
                            NeighborInfo_rel_pos,  # 2
                            NeighborInfo_rel_vel,  # 2
                            NeighborInfo_rel_heading,  # 1
                            NeighborInfo_boundingbox,  # 2
                        ], -1).astype(np.float32)

                        if len(on_road_neighbors) < 5:  # padding
                            NeighborInfo[len(on_road_neighbors):] = 0

                        obs.append(np.concatenate([
                            NeighborInfo.reshape(-1, ),  # (25)
                            EnvInfo.reshape(-1, ),  # (15)
                        ]))

                        # 确定action：相对速度小于0.2->0，不小于0.2->1
                        speed, heading, lane_index, lane_id = state.speed, state.heading, state.lane_index, state.lane_id
                        try:
                            speed_limit = current_raw_obs.waypoint_paths[lane_index][0].speed_limit
                            # 有时waymo数据集的lane_index会out of range
                        except:
                            speed_limit = current_raw_obs.waypoint_paths[0][0].speed_limit
                        if speed / speed_limit >= 0.2:
                            action = 1  # moving
                        else:
                            action = 0  # keeping still
                        actions.append(action)
                        # 确定terminal
                        next_raw_obs = vehicle_data[float(sim_time_next)]
                        events = next_raw_obs.events
                        # 使用next_obs判定是否到达目的地，若到达则terminal
                        next_position = vehicle_data[float(sim_time_next)].ego_vehicle_state.position
                        goal_reward = goal_region_reward(
                            threshold,
                            goal_pos_x,
                            goal_pos_y,
                            next_position[0],
                            next_position[1],
                        )  # 若到达终点则奖励+10，不确定是否要使用这一项
                        if float(sim_time) > 8.7:
                            aaa = 1
                        is_reached_goal = True if goal_reward > 0 else False
                        if (not (events.off_road or events.reached_goal or events.reached_max_episode_steps or
                                 is_reached_goal)) and len(events.collisions) == 0:
                            terminal = 0
                        else:
                            terminal = 1
                        terminals.append(terminal)
                        # 确定reward：[distance_travelled, action_reward, goal_reward]
                        dist_reward = next_raw_obs.distance_travelled
                        action_reward = {
                            0: 0,
                            1: 1
                        }[action]
                        reward = [dist_reward, action_reward, goal_reward]

                        rewards.append(reward)
                        episode_terminals.append(0)
                        if terminal == 1:
                            break
                    episode_terminals[-1] = 1  # 在每个episode的最后一个step，将episode_terminals的最后一个元素修改为1
                    print(f'ind {sc_index} scen {scenario} '
                          f'total_len {len(obs) - last_len} ter_sum {sum(episode_terminals)}')
                    last_len = len(obs)

            print(str(len(obs)) + " pieces of turning/changing lane data are added into dataset.")
            obs = np.array(obs)
            actions = np.array(actions)
            rewards = np.array(rewards)
            terminals = np.array(terminals)
            episode_terminals = np.array(episode_terminals)

            dataset = MDPDataset(obs, actions, rewards, terminals, episode_terminals=episode_terminals)
            return dataset, None

    except ImportError as e:
        raise ImportError(
            "waymo dataset is not prepared.\n"
            "get dataset from https://waymo.com/open/download/ and convert it to smarts type"
        ) from e

def get_dataset(env_name: str, type=None, input_dir=None) -> Tuple[MDPDataset, gym.Env]:
    """Returns dataset and envrironment by guessing from name.

    This function returns dataset by matching name with the following datasets.

    - cartpole-replay
    - cartpole-random
    - pendulum-replay
    - pendulum-random
    - d4rl-pybullet
    - d4rl-atari
    - d4rl

    .. code-block:: python

       import d3rlpy

       # cartpole dataset
       dataset, env = d3rlpy.datasets.get_dataset('cartpole')

       # pendulum dataset
       dataset, env = d3rlpy.datasets.get_dataset('pendulum')

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if env_name == "cartpole-replay":
        return get_cartpole(dataset_type="replay")
    elif env_name == "cartpole-random":
        return get_cartpole(dataset_type="random")
    elif env_name == "pendulum-replay":
        return get_pendulum(dataset_type="replay")
    elif env_name == "pendulum-random":
        return get_pendulum(dataset_type="random")
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(env_name)
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(env_name)
    elif re.match(re.compile("|".join(ATARI_GAMES)), env_name):
        return get_atari(env_name)
    elif env_name == "smarts_waymo":
        return get_waymo(env_name, type, input_dir)
    raise ValueError(f"Unrecognized env_name: {env_name}.")


