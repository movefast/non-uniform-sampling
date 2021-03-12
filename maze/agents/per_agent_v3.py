import agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from maze.buffer.per.prioritized_memory_v3 import Memory
from maze.replay_buffer import Transition
from mdp import autograd_hacks

criterion = torch.nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.tanh = nn.ReLU()
        # 2-layer nn
        self.i2h = nn.Linear(input_size, input_size//2+1, bias=False)
        self.h2o = nn.Linear(input_size//2+1, output_size, bias=False)
        # linear
        # self.i2o = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        # 2-layer nn
        x = self.i2h(x)
        x = self.tanh(x)
        x = self.h2o(x)
        # linear
        # x = self.i2o(x)
        return x


class LinearAgent(agent.BaseAgent):
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]

        self.num_meta_update = agent_init_info["num_meta_update"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.batch_size      = agent_init_info.get("batch_size", 10)
        self.buffer_size     = agent_init_info.get("buffer_size", 1000)

        self.per_alpha = agent_init_info.get("per_alpha", None)
        self.geo_alpha = agent_init_info.get("geo_alpha", None)
        self.buffer_beta = agent_init_info["buffer_beta"]
        self.beta_increment = agent_init_info.get("beta_increment", 0.001)

        self.correction = agent_init_info["correction"]
        self.recency_bias = agent_init_info.get("recency_bias", True)
        self.use_grad_norm = agent_init_info.get("grad_norm", False)
        self.tau = agent_init_info.get("tau", None)
        self.lam = agent_init_info.get("lam", None)
        self.min_weight = agent_init_info.get("min_weight", None)
        self.weighting_strat = agent_init_info["weighting_strat"]
        self.sim_mode = agent_init_info.get("sim_mode", 0)
        self.decay_by_uncertainty = agent_init_info.get("decay_by_uncertainty", None)

        self.nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.weights_init(self.nn)
        autograd_hacks.add_hooks(self.nn)
        self.target_nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        self.buffer = Memory(self.buffer_size, self.per_alpha, self.buffer_beta, self.beta_increment, self.weighting_strat, self.lam, self.tau, self.min_weight, self.geo_alpha, self.sim_mode, self.decay_by_uncertainty)
        self.tau = 0.5
        self.updates = 0

        self.sampled_state = np.zeros(self.num_states)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def get_state_feature(self, state):
        state, is_door = state
        state = np.eye(self.num_states)[state]
        state = torch.FloatTensor(state).to(device)[None, ...]
        return state

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        # Choose action using epsilon greedy.
        self.is_door = None
        self.feature = None
        state = self.get_state_feature(state)
        with torch.no_grad():
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps = 0
        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        # Choose action using epsilon greedy.
        state = self.get_state_feature(state)

        with torch.no_grad():
            current_q = self.nn(state)
        current_q.squeeze_()
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        # template from fpp_new
        # error = torch.abs(self.prev_action_value - reward - self.discount * target_q.max(1)[0]).item()
        # error = torch.abs(self.prev_action_value - reward - self.discount * current_q.max()).item()
        error = torch.abs(self.prev_action_value - reward - self.discount * current_q[action]).item()
        if self.recency_bias:
            self.buffer.add(self.buffer.maxp(), self.prev_state, self.prev_action, state, action, reward, self.discount)
        else:
            self.buffer.add(error, self.prev_state, self.prev_action, state, action, reward, self.discount)

        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1
        if len(self.buffer) > self.batch_size:
            self.batch_train()
        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        state = self.get_state_feature(state)
        if append_buffer:
            if self.recency_bias:
                self.buffer.add(self.buffer.maxp(), self.prev_state, self.prev_action, state, 0, reward, 0)
            else:
                error = torch.abs(self.prev_action_value - reward).item()
                self.buffer.add(error, self.prev_state, self.prev_action, state, 0, reward, 0)
        if len(self.buffer) > self.batch_size:
            self.batch_train()

    def batch_train(self):
        self.updates += 1
        self.nn.train()
        for _ in range(self.num_meta_update):
            transitions, idxs, is_weight = self.buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
            new_state_batch = torch.cat(batch.new_state)
            new_action_batch = torch.LongTensor(batch.new_action).view(-1, 1).to(device)
            reward_batch = torch.FloatTensor(batch.reward).to(device)
            discount_batch = torch.FloatTensor(batch.discount).to(device)
            # similarities = state_batch @ state_batch.T #/ state_action_batch.norm().pow(2)

            # scatter_idxs = torch.tensor(idxs).repeat(len(idxs), 1)
            # self.buffer.sims[idxs] = self.buffer.sims[idxs].scatter(1, scatter_idxs, similarities.to(dtype=torch.float64))


            self.sampled_state += state_batch.sum(0).detach().cpu().numpy()

            current_q = self.nn(state_batch)
            q_learning_action_values = current_q.gather(1, action_batch)
            with torch.no_grad():
                # ***
                # new_q = self.target_nn(new_state_batch)
                new_q = self.nn(new_state_batch)
            # max_q = new_q.max(1)[0]
            # max_q = new_q.mean(1)[0]
            max_q = new_q.gather(1, new_action_batch).squeeze_()
            target = reward_batch
            target += discount_batch * max_q
            target = target.view(-1, 1)

            # 1) correct with is weight
            if self.correction:
                temp = F.mse_loss(q_learning_action_values, target,reduction='none')
                loss = torch.Tensor(is_weight).to(device) @ temp
            # 2) no is correction
            else:
                loss = criterion(q_learning_action_values, target)

            if self.use_grad_norm:
                # 1) use autograd_hacks
                errors = [0] * self.batch_size
                self.optimizer.zero_grad()
                loss.backward()
                autograd_hacks.compute_grad1(self.nn)
                for param in self.nn.parameters():
                    for i in range(self.batch_size):
                        errors[i] += param.grad1[i].norm(2).item() ** 2
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                autograd_hacks.clear_backprops(self.nn)
                # if self.updates % 100 == 0:
                #     self.update()
                for i in range(self.batch_size):
                    self.buffer.update(idxs[i], np.power(errors[i]**.5, self.per_power))
            else:
                with torch.no_grad():
                    errors = torch.abs((q_learning_action_values - target).squeeze_(dim=-1))
                    # 1) batch
                    self.buffer.batch_update(idxs, errors)
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.nn.parameters():
                    param.grad.data.clamp_(-1, 1)
                # get grad similarities
                with torch.no_grad():
                    temp = list(self.nn.h2o.parameters())[0].grad1.sum(dim=1)
                    similarities = temp @ temp.T
                    scatter_idxs = torch.tensor(idxs).repeat(len(idxs), 1)
                    self.buffer.sims[idxs] = self.buffer.sims[idxs].scatter(1, scatter_idxs, similarities.to(dtype=torch.float64))

                self.optimizer.step()

                # if self.updates % 100 == 0:
                #     self.update()

    def update(self):
        # target network update
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def get_grad_norm(self):
        total_norm = 0
        for p in self.nn.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def update_target(self):
        self.target_nn.load_state_dict(self.nn.state_dict())

# old importance weight update code backup
# with torch.no_grad():
# errors = torch.abs((q_learning_action_values - target).squeeze_(dim=-1))
# for i in range(self.batch_size):
    # compare the loss term with grad norm
    # self.optimizer.zero_grad()
    # errors[i].backward(retain_graph=True)
    # print(self.get_grad_norm(), errors[i].item())
    # self.buffer.update(idxs[i], np.power(self.get_grad_norm(), self.per_power))
    # self.buffer.update(idxs[i], np.power(errors[i].item(), self.per_power))
