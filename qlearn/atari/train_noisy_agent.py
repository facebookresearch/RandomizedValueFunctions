# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import csv
import argparse
import time
import random
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter

from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.misc_util import boolean_flag, pretty_eta, RunningAvg
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from qlearn.atari.noisy_agent import AtariNoisyAgent


GAMES = ['BeamRiderNoFrameskip-v4',
         'SpaceInvadersNoFrameskip-v4',
         'BreakoutNoFrameskip-v4',
         'EnduroNoFrameskip-v4',
         'QbertNoFrameskip-v4',
         'SeaquestNoFrameskip-v4',
         'AlienNoFrameskip-v4',
         'AmidarNoFrameskip-v4',
         'FrostbiteNoFrameskip-v4',
         'TutankhamNoFrameskip-v4',
         'BankHeistNoFrameskip-v4',
         'AsterixNoFrameskip-v4',
         'GravitarNoFrameskip-v4']


SEEDS = [486, 750, 352,  93,  86]

RUN_ID = []
for seed in SEEDS:
    for game in GAMES:
        RUN_ID.append((seed, game))

def parse_args():
    parser = argparse.ArgumentParser("Noisy DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default='PongNoFrameskip-v4', help="name of game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    # parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate for RMSprop optimizer")
    # parser.add_argument("--alpha", type=float, default=0.95, help="alpha (squared gradient momentum) parameter for RMSprop optimizer")
    # parser.add_argument("--momentum", type=float, default=0.95, help="momentum parameter for RMSprop optimizer")
    # parser.add_argument("--eps-rmsprop", type=float, default=0.01, help="epsilon (min squared gradient) parameter for RMSprop optimizer")
    parser.add_argument("--num-steps", type=int, default=int(1e7),
                        help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=10000,
                        help="number of iterations between every target network update")
    parser.add_argument("--learning-starts", type=int, default=50000,
                        help="number of iterations after which learning starts")
    # boolean_flag(parser, "double-q", default=False, help="whether or not to use double q learning")
    parser.add_argument("--double-q", type=int, default=0, help="whether or not to use double q learning")
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="log_dir",
                        help="directory in which tensorboard events will be written out.")
    parser.add_argument("--save-dir", type=str, default="save_dir",
                        help="directory in which training state and model will be saved")
    parser.add_argument("--save-freq", type=int, default=int(1e6),
                        help="save model once every time this many iterations are completed")

    parser.add_argument("--final-exploration", type=float, default=0.1,
                        help="final value of epsilon in epsilon greedy exploration")
    parser.add_argument("--final-exploration-frame", type=int, default=int(1e6),
                        help="the number of frames over which the initial value of epsilon is linearly annealed to its final value")
    # New options
    parser.add_argument("--print-freq", type=int, default=100, help="printing frequency")

    parser.add_argument("--run-index", type=int, default=None, help="index RUN_ID")
    parser.add_argument("--cuda", type=int, default=1, help="whether or not to use cuda")
    parser.add_argument("--agent", type=str, default="NoisyDQN", help="which agent to run")
    parser.add_argument("--discount", type=float, default=0.99, help="discount factor")
    parser.add_argument("--model", type=str, default=None, help="model directory to load")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.run_index is not None:
        args.seed, args.env = RUN_ID[args.run_index]

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # Log
    date = time.strftime('%Y-%m-%d.%H%M')
    log_dir = '{}/{}-{}-seed-{}-{}'.format(args.log_dir, args.env, args.agent, args.seed, date)
    save_dir = '{}/{}-{}-seed-{}-{}'.format(args.save_dir, args.env, args.agent, args.seed, date)

    log = SummaryWriter(log_dir)
    print('Writing logs to {}'.format(log_dir))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # with open(save_dir + '/error_monitor.csv', "wt") as monitor_file:
    #     monitor = csv.writer(monitor_file)
    #     monitor.writerow(['update', 'error', str(int(args.num_steps / args.learning_freq))])

    with open(save_dir + '/reward_monitor.csv', "wt") as monitor_file:
        monitor = csv.writer(monitor_file)
        monitor.writerow(['epoch', 'reward', str(args.num_steps)])

    with open(save_dir + "/params.pkl", 'wb') as f:
        pickle.dump(args, f)

    # Create and seed the env.
    env = make_atari(args.env)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, scale=False)
    env.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # TODO
    num_actions = env.action_space.n
    agent = AtariNoisyAgent(args, env.observation_space.shape[-1], num_actions)

    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    start_time, start_steps = None, None
    steps_per_iter = RunningAvg(0.999)
    iteration_time_est = RunningAvg(0.999)
    obs = env.reset()
    num_iters = 0
    num_episodes = 0
    num_updates = 0
    prev_lives = None
    episode_rewards = [0.0]
    td_errors_list = []
    best_score = None

    while True:
        num_iters += 1
        # Take action and store transition in the replay buffer.
        if num_iters <= args.learning_starts:
            action = random.randrange(num_actions)
        else:
            # Reshape state to (1, channels, x_dim, y_dim)
            action = agent.act(np.transpose(np.array(obs)[None], [0, 3, 1, 2]), eval=False)
        # import pdb
        # pdb.set_trace()
        new_obs, rew, done, info = env.step(action)
        death = done or (prev_lives is not None and info['ale.lives'] < prev_lives and info['ale.lives'] > 0)
        prev_lives = info['ale.lives']

        replay_buffer.add(obs, action, np.sign(rew), new_obs, float(death))
        obs = new_obs
        episode_rewards[-1] += rew

        if done:
            log.add_scalar('reward', episode_rewards[-1], num_iters)
            episode_rewards.append(0.0)
            obs = env.reset()
            num_episodes += 1

        if num_iters > args.learning_starts and num_iters % args.learning_freq == 0:

            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
            # Reshape state to (batch, channels, x_dim, y_dim)
            obses_t = np.transpose(obses_t, [0, 3, 1, 2])
            obses_tp1 = np.transpose(obses_tp1, [0, 3, 1, 2])

            # TODO
            td_errors = agent.learn(obses_t, actions, rewards, obses_tp1, dones)
            td_errors_list.append(td_errors.item())
            log.add_scalar('td_error', td_errors.item(), num_iters)

            num_updates += 1

            # Update target network.
        if num_iters > args.learning_starts and num_iters % args.target_update_freq == 0:
            # TODO
            agent.update_target_net()

        if start_time is not None:
            steps_per_iter.update(num_iters - start_steps)
            iteration_time_est.update(time.time() - start_time)
        start_time, start_steps = time.time(), num_iters

        if num_iters > args.num_steps:
            break

        if done and num_episodes % args.print_freq == 0 and num_episodes >= args.print_freq:
            steps_left = args.num_steps - num_iters
            completion = np.round(num_iters / args.num_steps, 1)
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            logger.record_tabular("% completion", completion)
            logger.record_tabular("total steps", num_iters)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("reward (100 epi mean)", mean_100ep_reward)

            fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                            if steps_per_iter._value is not None else "calculating...")
            logger.record_tabular("FPS", fps_estimate)
            logger.dump_tabular()
            logger.log()
            logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
            logger.log()

            with open(save_dir + '/reward_monitor.csv', "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([num_iters, mean_100ep_reward])

            # if len(td_errors_list) > 0:
            #     with open(save_dir + '/error_monitor.csv', "a") as monitor_file:
            #         monitor = csv.writer(monitor_file)
            #         monitor.writerow([num_updates, round(np.mean(td_errors_list), 4)])

            if best_score is None or mean_100ep_reward > best_score:
                logger.log("Saving model due to mean reward increase: {} -> {}".format(
                               best_score, mean_100ep_reward))
                best_score = mean_100ep_reward
                torch.save(agent.online_net.state_dict(), log_dir + '/best_model.torch')

            torch.save(agent.online_net.state_dict(), save_dir + '/current_model.torch')
            # td_errors_list = [0.0]
