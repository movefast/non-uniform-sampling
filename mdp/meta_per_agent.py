from copy import deepcopy

import agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mdp.prioritized_memory import Memory
from mdp.replay_buffer import Transition

criterion = torch.nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.tanh = nn.ReLU()
        # self.i2h = nn.Linear(input_size, input_size//2+1, bias=False)
        # self.h2o = nn.Linear(input_size//2+1, output_size, bias=False)
        self.i2o = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        # x = self.i2h(x)
        # x = self.tanh(x)
        # x = self.h2o(x)
        x = self.i2o(x)
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
        self.meta_step_size = agent_init_info["meta_step_size"]

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.batch_size      = agent_init_info.get("batch_size", 10)
        self.buffer_size     = agent_init_info.get("buffer_size", 1000)

        self.buffer_alpha = agent_init_info["buffer_alpha"]
        self.buffer_beta = agent_init_info["buffer_beta"]
        self.beta_increment = agent_init_info.get("beta_increment", 0.001)

        self.online_opt = agent_init_info.get("online_opt")

        self.correction = agent_init_info["correction"]

        self.nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.weights_init(self.nn)
        self.target_nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.update_target()
        
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        if self.online_opt == "sgd":
            self.online_optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.step_size)
        elif self.online_opt == "adam":
            self.online_optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        else:
            raise NotImplementedError("you chose an optimizer other than sgd and adam")
        self.buffer = Memory(self.buffer_size, self.buffer_alpha, self.buffer_beta, self.beta_increment)
        self.tau = 0.5
        self.updates = 0

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def get_state_feature(self, state):
        state, is_door = state
        state = np.eye(self.num_states)[state]
        state = torch.Tensor(state).to(device)[None, ...]
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

        self.buffer.add(self.buffer.maxp(), self.prev_state, self.prev_action, state, action, reward, self.discount)
        
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
            self.buffer.add(self.buffer.maxp(), self.prev_state, self.prev_action, state, 0, reward, 0)
        if len(self.buffer) > self.batch_size:
            self.batch_train()

    def batch_train(self):
        self.updates += 1
        self.nn.train()
        transitions, idxs, is_weight = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        new_state_batch = torch.cat(batch.new_state)
        new_action_batch = torch.LongTensor(batch.new_action).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        discount_batch = torch.FloatTensor(batch.discount).to(device)

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
        errors = torch.abs((q_learning_action_values - target).squeeze_(dim=-1))
        for i in range(self.batch_size):
            self.buffer.update(idxs[i], np.power(errors[i].item(), self.per_power))

        weights_before = deepcopy(self.nn.state_dict())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        weights_after = self.nn.state_dict()
        self.nn.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.meta_step_size) for name in weights_before})
        # if self.updates % 100 == 0:
        #     self.update()

        transitions, _, _, _ = self.buffer.last_n(1)
        # transitions = self.buffer.last_n(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        new_state_batch = torch.cat(batch.new_state)
        new_action_batch = torch.LongTensor(batch.new_action).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        discount_batch = torch.FloatTensor(batch.discount).to(device)

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
        loss = criterion(q_learning_action_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.online_optimizer.step()

    def update(self):
        # target network update
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def update_target(self):
        self.target_nn.load_state_dict(self.nn.state_dict())
