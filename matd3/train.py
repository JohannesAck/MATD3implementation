import argparse
import pickle
import time
from copy import deepcopy

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from matd3.trainer.matd3 import  MATD3AgentTrainer
from multiagent.environment import MultiAgentEnv

logger = None

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="matd3", help="policy for good agents (matd3 or maddpg)")
    parser.add_argument("--adv-policy", type=str, default="matd3", help="policy of adversaries (matd3 or maddpg)")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--update-rate", type=int, default=100, help="after this many steps the critic is trained")
    parser.add_argument("--policy-update-rate", type=int, default=2,
                        help="after this many critic updates the target networks and policy are trained")
    parser.add_argument("--use-critic-noise", action="store_true", default=False, help="use noise in critic update next action")
    parser.add_argument("--use-critic-noise-self", action="store_true", default=False, help="use noise in critic update next action")
    parser.add_argument("--critic-action-noise-stddev", type=float, default=0.2)
    parser.add_argument("--action-noise-clip", type=float, default=0.5)
    parser.add_argument("--critic-zero-if-done", action="store_true", default=False, help="set q value to zero in critic update after done")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='def_exp_name', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--real-q-log", action="store_true", default=False,help="Evaluates approx. real q value after every 5 save-rates")
    parser.add_argument("--q-log-ep-len", type=int, default=200, help="Number of steps per state in q_eval")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False, help="Saves all locations and termination locations")
    parser.add_argument("--benchmark-iters", type=int, default=10000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--record-episodes", action="store_true", default=False, help="save rgb arrays of episodes")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def calculate_real_q_value(env: MultiAgentEnv, agents, world_state_buffer, action_n_buffer, start_episode_step_buffer,
                           obs_n_buffer, num_start_states, args):
    """

    :param env:
    :param agents:
    :param world_state_buffer: buffer of world states, from which we randomly sample
    :param action_n_buffer: buffer of action chosen in the world_state of same index
    :param num_start_states:
    :param len_eval:
    :return:
    """
    world_sample_indexes = np.random.choice(range(len(world_state_buffer)), num_start_states)
    discounted_run_rewards_n = []
    q_values_n = []
    for start_idx, world_idx in enumerate(world_sample_indexes):
        env.world = deepcopy(world_state_buffer[world_idx])
        episode_reward_n = []
        action_n = action_n_buffer[world_idx]
        obs_n, reward_n, done_n, info_n = env.step(action_n)
        episode_reward_n.append(reward_n)
        # if arglist.q_log_full_episodes:
        episode_step = 0
        # else:
        #     episode_step = start_episode_step_buffer[world_idx]

        terminal = False
        obs_n_reshaped = []
        action_n_reshaped = []
        for ag_idx in range(len(obs_n)):
            obs_n_reshaped.append([obs_n[ag_idx]])
            action_n_reshaped.append([action_n[ag_idx]])
        q_values_n.append([agent.q_debug['q_values'](*(obs_n_reshaped + action_n_reshaped)) for agent in agents])

        while not (all(done_n) or terminal):
            action_n = [agent.action(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, reward_n, done_n, info_n = env.step(action_n)
            episode_reward_n.append(reward_n)

            terminal = episode_step >= arglist.q_log_ep_len
            episode_step += 1

        discount_factors = np.power(args.gamma, np.arange(0, len(episode_reward_n), dtype=np.int))
        discounted_run_rewards_n.append(np.dot(discount_factors, np.array(episode_reward_n)))

    q_mean = np.mean(q_values_n, 0)[:,0]
    real_mean = np.mean(discounted_run_rewards_n, 0)
    return q_mean, real_mean




def get_trainers(env, num_adversaries, obs_shape_n, arglist, good_agent_mode='matd3', adv_agent_mode='matd3'):
    trainers = []
    model = mlp_model
    if good_agent_mode=='matd3':
        good_trainer = MATD3AgentTrainer
    elif good_agent_mode=='maddpg':
        good_trainer = MADDPGAgentTrainer
    else:
        raise RuntimeError('Unknown agent mode specified' + str(good_agent_mode))
    if adv_agent_mode== 'matd3':
        adv_trainer = MATD3AgentTrainer
    elif adv_agent_mode== 'maddpg':
        adv_trainer= MADDPGAgentTrainer
    else:
        raise RuntimeError('Unknown agent mode specified' + str(adv_agent_mode))

    for i in range(num_adversaries):
        trainers.append(adv_trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(good_trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train_maddpg(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist,
                                good_agent_mode=arglist.good_policy, adv_agent_mode=arglist.adv_policy)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver(max_to_keep=None)
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        if arglist.real_q_log:
            world_state_buffer, action_n_buffer, start_episode_step_buffer, obs_n_buffer = [], [], [], []
            q_means, real_means = [], []

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)  # note: unused, never happens
            terminal = (episode_step >= arglist.max_episode_len)
            done = done or terminal

            if arglist.real_q_log:
                world_state_buffer.append(deepcopy(env.world))
                obs_n_buffer.append(obs_n)
                action_n_buffer.append(action_n)
                start_episode_step_buffer.append(episode_step)

            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done, terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew



            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)  # add element for next episode
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            for agent in trainers:
                loss = agent.update(trainers, train_step)


            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                if arglist.save_dir != '/tmp/policy/':
                    U.save_state(arglist.save_dir + arglist.exp_name, saver=saver, global_step=len(episode_rewards))
                else:
                    U.save_state(arglist.save_dir, saver=saver)                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:-1]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:-1]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:-1]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:-1]))

                if arglist.real_q_log and (len(episode_rewards) % (5 * arglist.save_rate) == 0):
                    q_mean, real_mean = calculate_real_q_value(deepcopy(env), trainers,
                                                               world_state_buffer=world_state_buffer,
                                                               action_n_buffer=action_n_buffer,
                                                               obs_n_buffer=obs_n_buffer,
                                                               start_episode_step_buffer=start_episode_step_buffer,
                                                               num_start_states=200,
                                                               args=arglist)
                    world_state_buffer, action_n_buffer, start_episode_step_buffer, obs_n_buffer = [], [], [], []
                    q_means.append(q_mean)
                    real_means.append(real_mean)
                    print('Q-mean: ' + str(q_mean) + ' Real mean: ' + str(real_mean))




            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                args_file_name = arglist.plots_dir + arglist.exp_name + '_args.pkl'
                with open(args_file_name, 'wb') as fp:
                    pickle.dump(arglist, fp)
                if arglist.real_q_log:
                    real_q_path = arglist.plots_dir + arglist.exp_name + '_q_values.pkl'
                    with open(real_q_path, 'wb') as fp:
                        pickle.dump({'q_means': q_means, 'real_means': real_means}, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train_maddpg(arglist)
