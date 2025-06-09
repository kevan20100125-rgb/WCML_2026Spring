import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Environment_marl_test
import os
from replay_memory import ReplayMemory
from torch.distributions import Categorical
from torch.distributions.normal import Normal
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]]
width = 750/2
height = 1298/2
label = 'marl_model_4_1'
label_sarl = 'sarl_model_hppo_4_1'
n_veh = 4
n_neighbor = 1
n_RB = n_veh
# Environment Parameters
env = Environment_marl_test.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()   # initialize parameters in env
n_episode = 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4
n_episode_test = 100  # test episodes
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
######################################################


def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


def get_state_b(env, idx=(0, 0), ind_episode=1.):
    """ Get state from the environment """

    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :]
                - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode])))


# For Actorp
def get_state_p(env, idx=(0, 0), ind_episode=1., band=0.):
    """ Get state from the environment """

    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :]
                - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode, band])))


band_input_size = len(get_state_b(env=env))
power_input_size = len(get_state_p(env=env))


n_input_size = len(get_state(env=env))
n_output_size = n_RB * len(env.V2V_power_dB_List)


class DQN(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
        super(DQN, self).__init__()
        self.fc_1 = nn.Linear(input_size, n_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(n_hidden3, output_size)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        return x


class Agent:
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = False
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)
        self.model = DQN(n_input_size, n_hidden_1, n_hidden_2, n_hidden_3, n_output_size).to(device)
        # if torch.cuda.device_count()>1:
        #    self.model = nn.DataParallel(self.model)
        # self.model.to(device)
        self.target_model = DQN(n_input_size, n_hidden_1, n_hidden_2, n_hidden_3, n_output_size).to(device)  # Target Model
        self.target_model.eval()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001, momentum=0.05, eps=0.01)
        self.loss_func = nn.MSELoss()

    def predict(self, s_t, ep=0., test= False):
        n_power_levels = len(env.V2V_power_dB_List)
        # state_t = torch.from_numpy(s_t).type(torch.float32).view([1, self.memory_entry_size])
        if np.random.rand() < ep and not test:
            return np.random.randint(n_RB * n_power_levels)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device))
                return q_values.max(1)[1].item()

    def predict_sarl(self, s_t):
        with torch.no_grad():
            q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device))
            return q_values.max(1)[1].item()

    def save_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.model.state_dict(), model_path+'.ckpt')
        torch.save(self.target_model.state_dict(), model_path+'_t.ckpt')

    def load_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        self.model.load_state_dict(torch.load(model_path + '.ckpt'))
        self.target_model.load_state_dict(torch.load(model_path + '_t.ckpt'))


class Actorb(nn.Module):  # Actor for choosing band
    def __init__(self):
        super(Actorb, self).__init__()
        self.fc_1 = nn.Linear(band_input_size, 500)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(250, 120)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(120, n_RB)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        action_prob = F.softmax(self.fc_4(x), dim=1)
        return action_prob


class Criticb(nn.Module):
    def __init__(self):
        super(Criticb, self).__init__()
        self.fc_1 = nn.Linear(band_input_size, 500)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(250, 120)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(120, 1)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x


class Actorp(nn.Module):  # Actor for choosing power
    def __init__(self):
        super(Actorp, self).__init__()
        self.fc_1 = nn.Linear(power_input_size, 500)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(250, 120)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.mu = nn.Linear(120, 1)
        self.mu.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        mu = torch.tanh(self.mu(x)) * 100 + 100
        return mu


class Criticp(nn.Module):
    def __init__(self):
        super(Criticp, self).__init__()
        self.fc_1 = nn.Linear(power_input_size, 500)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(250, 120)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(120, 1)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x


class Agentb:
    def __init__(self):
        self.LAMBDA = 0.95
        self.discount = 0.99
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.K_epoch = 8

        self.actor = Actorb().to(device)
        self.critic = Criticb().to(device)

        self.loss_func = nn.MSELoss()
        self.data_buffer = []  # To store the experience
        self.counter = 0  # the number of experience tuple in data_buffer

    def choose_action(self, s_t):
        #  Return the action, and the probability to choose this action
        s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor(s_t)
        c = Categorical(action_prob)
        action = c.sample()
        a_log_prob = action_prob[:, action.item()]
        return action.item(), a_log_prob.item()

    def load_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        self.actor.load_state_dict(torch.load(model_path + '_ab.ckpt'))
        self.critic.load_state_dict(torch.load(model_path + '_cb.ckpt'))


class Agentp:
    def __init__(self):
        self.LAMBDA = 0.95
        self.discount = 0.99
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.K_epoch = 8
        self.sigma = 0.001

        self.actor = Actorp().to(device)
        self.critic = Criticp().to(device)

        self.loss_func = nn.MSELoss()
        self.data_buffer = []  # To store the experience
        self.counter = 0  # the number of experience tuple in data_buffer

    def choose_action(self, s_t):
        #  Return the action, and the probability to choose this action
        s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mu = self.actor(s_t)
        c = Normal(mu, self.sigma)
        action = torch.clamp(c.sample(), 0, 200.0)   # power in [0, 200]mW
        a_log_prob = c.log_prob(action)
        return action.item(), a_log_prob.item()

    def load_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        self.actor.load_state_dict(torch.load(model_path + '_ap.ckpt'))
        self.critic.load_state_dict(torch.load(model_path + '_cp.ckpt'))


# -----------------------------------------------------------------------------------------------------
agents = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)
agent_sarl_b = Agentb()
agent_sarl_p = Agentp()
# -----------------------------------Testing----------------------------------------------------------
print("\nRestoring the model...")
for i in range(n_veh):
    for j in range(n_neighbor):
        model_path = label + '/agent_' + str(i * n_neighbor + j)
        agents[i * n_neighbor + j].load_models(model_path)
    # restore the single-agent model
    model_path_single = label_sarl + '/agent'
    agent_sarl_b.load_models(model_path_single)
    agent_sarl_p.load_models(model_path_single)

    V2I_rate_list = []
    V2V_success_list = []

    V2I_rate_list_rand = []
    V2V_success_list_rand = []

    V2I_rate_list_sarl = []
    V2V_success_list_sarl = []

    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_sarl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    demand_sarl = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])
    demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode+1, n_veh, n_neighbor])

    action_all_band = np.zeros([n_veh, n_neighbor], dtype='int32')
    action_all_power = np.zeros([n_veh, n_neighbor], dtype='float64')
    action_all_testing_dpra = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

with torch.no_grad():
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        env.demand_sarl = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit_sarl = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links_sarl = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        V2I_rate_per_episode = []
        V2I_rate_per_episode_rand = []
        V2I_rate_per_episode_sarl = []

        for test_step in range(n_step_per_episode):
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action = agents[i * n_neighbor + j].predict(state_old, epsi_final, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()
            V2I_rate, V2V_success, V2V_rate = env.act_for_testing(action_temp)
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps

            rate_marl[idx_episode, test_step, :, :] = V2V_rate
            demand_marl[idx_episode, test_step + 1, :, :] = env.demand

            # random baseline
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor])  # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor])  # power
            V2I_rate_rand, V2V_success_rand, V2V_rate_rand = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps

            rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand
            demand_rand[idx_episode, test_step + 1, :, :] = env.demand_rand

            # SARL
            remainder = test_step % (n_veh * n_neighbor)
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor
            state_sarl = get_state_b(env, [i, j], 1)
            band, _ = agent_sarl_b.choose_action(state_sarl)
            state_sarl_p = get_state_p(env, [i, j], 1, band)
            power, _ = agent_sarl_p.choose_action(state_sarl_p)
            action_all_band[i, j] = band  # chosen RB
            action_all_power[i, j] = 10 * np.log10(power + 1e-10)
            # action_all_power[i, j] = 10 * np.log10(power + 1e-10)  # power level
            action_band_temp = action_all_band.copy()
            action_power_temp = action_all_power.copy()
            V2I_rate_sarl, V2V_success_sarl, V2V_rate_sarl = env.act_for_testing_sarl(action_band_temp, action_power_temp)
            V2I_rate_per_episode_sarl.append(np.sum(V2I_rate_sarl))  # sum V2I rate in bps
            rate_sarl[idx_episode, test_step, :, :] = V2V_rate_sarl
            demand_sarl[idx_episode, test_step + 1, :, :] = env.demand_sarl
            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            env.Compute_Interference_sarl(action_band_temp, action_power_temp)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_rand)
                V2V_success_list_sarl.append(V2V_success_sarl)

        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))
        V2I_rate_list_sarl.append(np.mean(V2I_rate_per_episode_sarl))

        print('marl', round(np.average(V2I_rate_per_episode), 2), 'sarl',
              round(np.average(V2I_rate_per_episode_sarl), 2), 'rand', round(np.average(V2I_rate_per_episode_rand), 2))
        print('marl', V2V_success_list[idx_episode], 'sarl', V2V_success_list_sarl[idx_episode], 'rand',
              V2V_success_list_rand[idx_episode])
print('-------- marl -------------')
print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
print('Sum V2I rate:', round(np.average(V2I_rate_list), 2), 'Mbps')
print('Pr(V2V success):', round(np.average(V2V_success_list), 4))

print('-------- sarl -------------')
print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
print('Sum V2I rate:', round(np.average(V2I_rate_list_sarl), 2), 'Mbps')
print('Pr(V2V success):', round(np.average(V2V_success_list_sarl), 4))

print('-------- random -------------')
print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 2), 'Mbps')
print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))

current_dir = os.path.dirname(os.path.realpath(__file__))
marl_path = os.path.join(current_dir, 'model/' + label + '/rate_marl.mat')
scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
sarl_path = os.path.join(current_dir, 'model/' + label + '/rate_sarl.mat')
scipy.io.savemat(sarl_path, {'rate_sarl': rate_sarl})
rand_path = os.path.join(current_dir, 'model/' + label + '/rate_rand.mat')
scipy.io.savemat(rand_path, {'rate_rand': rate_rand})
demand_marl_path = os.path.join(current_dir, 'model/' + label + '/demand_marl.mat')
scipy.io.savemat(demand_marl_path, {'demand_marl': demand_marl})
demand_sarl_path = os.path.join(current_dir, 'model/' + label + '/demand_sarl.mat')
scipy.io.savemat(demand_sarl_path, {'demand_sarl': demand_sarl})
demand_rand_path = os.path.join(current_dir, 'model/' + label + '/demand_rand.mat')
scipy.io.savemat(demand_rand_path, {'demand_rand': demand_rand})
