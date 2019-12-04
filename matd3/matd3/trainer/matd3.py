import numpy as np
import tensorflow as tf

import common.tf_util as U
from maddpg import AgentTrainer
from common.distributions import make_pdtype
from maddpg.trainer.replay_buffer import ReplayBuffer


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, act_space_n, agent_idx, p_func, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None):
    """

    :param make_obs_ph_n:
    :param act_space_n:
    :param agent_idx:
    :param p_func: in base maddpg code = mlp_model
    :param q_func: in base maddpg code = mlp_model
    :param optimizer:
    :param grad_norm_clipping:
    :param local_q_func:
    :param num_units:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = [tf.layers.flatten(obs_ph) for obs_ph in make_obs_ph_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[agent_idx]

        p = p_func(p_input, int(act_pdtype_n[agent_idx].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[agent_idx].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[agent_idx] = act_pd.sample() #act_pd.mode() #
        q_input = tf.concat(obs_ph_n + act_input_n, 1)

        q = q_func(q_input, 1, scope="q_func" + str(1), reuse=True, num_units=num_units)[:,0]

        loss = -tf.reduce_mean(q)  + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=make_obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[make_obs_ph_n[agent_idx]], outputs=act_sample)
        p_values = U.function([make_obs_ph_n[agent_idx]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[agent_idx].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[agent_idx].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[make_obs_ph_n[agent_idx]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, agent_idx, q_func, q_function_idx, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = [tf.layers.flatten(obs_ph) for obs_ph in make_obs_ph_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[agent_idx], act_ph_n[agent_idx]], 1)
        q = q_func(q_input, 1, scope="q_func" + str(q_function_idx), num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func" + str(q_function_idx)))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=make_obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(make_obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func" + str(q_function_idx), num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func" + str(q_function_idx)))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(make_obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MATD3AgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train1, self.q_update1, self.q_debug1 = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            agent_idx=agent_index,
            q_function_idx=1,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.q_train2, self.q_update2, self.q_debug2 = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            agent_idx=agent_index,
            q_func=model,
            q_function_idx=2,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            agent_idx=agent_index,
            p_func=model,
            q_func=model,  #MLPmodel()
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.min_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        a = tf.summary.FileWriter("logdirMaddpg", tf.get_default_graph())
        a.flush()
        a.close()

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    @property
    def q_debug(self):
        return self.q_debug1

    def update(self, agents, train_step):
        if len(self.replay_buffer) < self.min_replay_buffer_len:  # replay buffer is not large enough
            return

        if not train_step % self.args.update_rate == 0:
            return


        self.replay_sample_index = self.replay_buffer.generate_sample_indices(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
        if self.args.use_critic_noise:
            for agent_idx in range(self.n):
                noise = np.random.normal(0, self.args.critic_action_noise_stddev, size=target_act_next_n[agent_idx].shape)
                clipped_noise = np.clip(noise, -self.args.action_noise_clip, self.args.action_noise_clip)
                target_act_next_n[agent_idx] = (target_act_next_n[agent_idx] + clipped_noise).tolist()
        elif self.args.use_critic_noise_self:
            noise = np.random.normal(0, self.args.critic_action_noise_stddev,
                                     size=target_act_next_n[self.agent_index].shape)
            clipped_noise = np.clip(noise, -self.args.action_noise_clip, self.args.action_noise_clip)
            target_act_next_n[self.agent_index] = target_act_next_n[self.agent_index] + clipped_noise
            target_act_next_n = target_act_next_n.tolist()
        else:
            target_act_next_n = target_act_next_n
        target_q_next1 = self.q_debug1['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q_next2 = self.q_debug2['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q_next = np.min([target_q_next1, target_q_next2], 0)
        if self.args.critic_zero_if_done:
            done_cond = done == True
            target_q_next[done_cond] = 0

        target_q = rew + self.args.gamma * target_q_next
        q_loss = self.q_train1(*(obs_n + act_n + [target_q]))
        q_loss = self.q_train2(*(obs_n + act_n + [target_q]))

        # train p network
        if train_step % (self.args.update_rate * self.args.policy_update_rate) == 0:
            p_loss = self.p_train(*(obs_n + act_n))
            self.p_update()
            self.q_update1()
            self.q_update2()

        # print('Agent' + str(self.agent_index)  + ' Qloss = ' + str(q_loss) + ' Ploss = ' + str(p_loss))
        # print('Replay buffer size:' + str(len(self.replay_buffer)))


        return [q_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
