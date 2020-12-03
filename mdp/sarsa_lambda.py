import agent
import numpy as np

TRACE_TYPE = ["accumulate", "replacing"]
# [Graded]
# Q-Learning agent here
class QLearningAgent(agent.BaseAgent):
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
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.lam = agent_init_info["lambda"]
        # default trace type to replacing
        # self.trace_type = TRACE_TYPE[agent_init_info.get("trace_type", 1)]
        
        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        
    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        self.z = np.zeros((self.num_states, self.num_actions))
        state = state[0]
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
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
        state = state[0]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        delta = self.step_size * (reward + self.discount * self.q[state, action] - self.q[self.prev_state, self.prev_action])
        self.z *= self.discount * self.lam
        self.z[self.prev_state, self.prev_action] += 1
        # self.q[self.prev_state, self.prev_action] += self.step_size * delta * self.z[self.prev_state, self.prev_action]
        self.q += self.step_size * delta * self.z
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
        # self.q[self.prev_state, self.prev_action] += self.step_size * (reward - self.q[self.prev_state, self.prev_action])
        delta = self.step_size * (reward - self.q[self.prev_state, self.prev_action])
        self.z *= self.discount * self.lam
        self.z[self.prev_state, self.prev_action] += 1
        self.q += self.step_size * delta * self.z
