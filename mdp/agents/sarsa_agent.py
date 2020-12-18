import agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdp.replay_buffer import ReplayMemory, Transition

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

        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        self.nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.weights_init(self.nn)
        self.target_nn = SimpleNN(self.num_states, self.num_actions).to(device)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.step_size)
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

        self.batch_train(self.prev_state, self.prev_action, state, action, reward, self.discount)
        
        self.prev_action_value = current_q[action]
        self.prev_state = state
        self.prev_action = action
        self.steps += 1
        
        return action

    def agent_end(self, reward, state, append_buffer=True):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        state = self.get_state_feature(state)
        if append_buffer:
            self.batch_train(self.prev_state, self.prev_action, state, 0, reward, 0)

    def batch_train(self, state, action, new_state, new_action, reward, discount):
        self.updates += 1
        self.nn.train()
        batch = Transition([state], [action], [new_state], [new_action], [reward], [discount])
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
        loss = criterion(q_learning_action_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # if self.updates % 100 == 0:
        #     self.update()

    def update(self):
        # target network update
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def update_target(self):
        self.target_nn.load_state_dict(self.nn.state_dict())
