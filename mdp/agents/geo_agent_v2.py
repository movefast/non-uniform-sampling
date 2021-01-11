import agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdp.buffer.geo_replay_buffer_v2 import ReplayMemory
from mdp.replay_buffer import Transition

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

        self.buffer_alpha = agent_init_info.get("buffer_alpha", None)
        self.buffer_beta = agent_init_info["buffer_beta"]
        self.beta_increment = agent_init_info.get("beta_increment", 0.001)

        self.correction = agent_init_info["correction"]
        self.tau_1 = agent_init_info.get("tau_1", None)
        self.tau_2 = agent_init_info.get("tau_2", None)
        self.lam = agent_init_info.get("lam", None)
        self.min_weight = agent_init_info["min_weight"]

        self.weighting_strat = agent_init_info["weighting_strat"] 
        # self.use_discor_loss = agent_init_info["use_discor_loss"]
        self.nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.weights_init(self.nn)
        self.target_nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
        self.buffer = ReplayMemory(self.buffer_size, self.buffer_alpha, self.buffer_beta, self.beta_increment, self.weighting_strat, self.lam, self.tau_1, self.tau_2, self.min_weight)
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

        self.buffer.add(self.prev_state, self.prev_action, state, action, reward, self.discount)
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
            self.buffer.add(self.prev_state, self.prev_action, state, 0, reward, 0)
        if len(self.buffer) > self.batch_size:
            self.batch_train()

    def batch_train(self):
        self.updates += 1
        self.nn.train()
        for _ in range(self.num_meta_update):
            transitions, idxs, is_weight = self.buffer.sample_geo(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
            new_state_batch = torch.cat(batch.new_state)
            new_action_batch = torch.LongTensor(batch.new_action).view(-1, 1).to(device)
            reward_batch = torch.FloatTensor(batch.reward).to(device)
            discount_batch = torch.FloatTensor(batch.discount).to(device)

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
                # integrate discor_weights into is_weight
                # if self.discor_correction:
                #     discor_weights = np.exp(-self.buffer.loss_weights[idxs] * self.discount / self.discor_tau) 
                #     is_weight *= discor_weights
                temp = F.mse_loss(q_learning_action_values, target,reduction='none')
                loss = torch.Tensor(is_weight).to(device) @ temp
            # 2) no is correction
            else:
                loss = criterion(q_learning_action_values, target)
            # discor_weights = np.exp(-self.buffer.loss_weights[idxs] * self.discount / self.tau_2) 
            errors = torch.abs((q_learning_action_values - target).squeeze_(dim=-1))

            # update loss weights for Discor correction
            nonzero_idxes = idxs != 0
            t_1_idxs = idxs-1
            self.buffer.loss_weights[idxs[nonzero_idxes]] = errors[nonzero_idxes].detach().cpu().numpy() + discount_batch.numpy()[nonzero_idxes] * self.buffer.loss_weights[t_1_idxs[nonzero_idxes]]

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            if self.weighting_strat != 3: 
                with torch.no_grad():
                    self.buffer.geo_weights += errors.mean().item() ** self.buffer_alpha
            # if self.updates % 10 == 0:
            #     self.update()

    def update(self):
        # target network update
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def update_target(self):
        self.target_nn.load_state_dict(self.nn.state_dict())
