import numpy as np
import random


class SimpleMDP():
    def __init__(self):
        super(SimpleMDP, self).__init__()

        # Define the action space (only one action)
        self.action_space =[0,1]

        # Define the observation space (states 1, 2, 3, 4)
        self.observation_space = [0,1,2,3,4,5,6,7,8]

        # Define transition probabilities P(s'|s, a)
        self.transitions = {
            0: {0: [(0,0), (1, 0.1), (2,0), (3, 0.3), (4, 0.1), (5, 0.2), (6,0.2), (7,0.05), (8,0.05)],
                1: [(0,0), (1, 0.05), (2,0), (3, 0.6), (4, 0.05), (5, 0.1), (6,0.1), (7,0.05), (8,0.05)]},
            1: {0: [(0,0.04), (1, 0), (2,0.06), (3, 0.2), (4, 0), (5, 0.2), (6,0.1), (7,0.2), (8,0.2)],
                1: [(0,0.1), (1, 0), (2,0.1), (3, 0.3), (4, 0), (5, 0.1), (6,0.1), (7,0.1), (8,0.2)]},
            2: {0: [(0,0), (1, 0.11), (2,0), (3, 0.3), (4, 0.09), (5, 0.19), (6,0.21), (7,0.04), (8,0.06)],
                1: [(0,0), (1, 0.04), (2,0), (3, 0.6), (4, 0.06), (5, 0.09), (6,0.11), (7,0.04), (8,0.06)]},
            3: {0: [(0,0.1), (1, 0.2), (2,0.1), (3, 0), (4, 0.2), (5, 0.04), (6,0.06), (7,0.1), (8,0.2)],
                1: [(0,0.04), (1, 0.1), (2,0.06), (3, 0), (4, 0.1), (5, 0.2), (6,0.1), (7,0.3), (8,0.1)]},
            4: {0: [(0,0.05), (1, 0), (2,0.05), (3, 0.2), (4, 0), (5, 0.19), (6,0.11), (7,0.19), (8,0.21)],
                1: [(0,0.09), (1, 0), (2,0.11), (3, 0.3), (4, 0), (5, 0.09), (6,0.11), (7,0.09), (8,0.21)]},
            5: {0: [(0,0.1), (1, 0.1), (2,0.1), (3, 0.1), (4, 0.2), (5, 0), (6,0), (7,0.2), (8,0.2)],
                1: [(0,0.04), (1, 0.1), (2,0.06), (3, 0.3), (4, 0.1), (5, 0), (6,0), (7,0.2), (8,0.2)]},
            6: {0: [(0,0.09), (1, 0.09), (2,0.11), (3, 0.1), (4, 0.21), (5, 0), (6,0), (7,0.19), (8,0.21)],
                1: [(0,0.03), (1, 0.09), (2,0.07), (3, 0.3), (4, 0.11), (5, 0), (6,0), (7,0.19), (8,0.21)]},
            7: {0: [(0,0.1), (1, 0.2), (2,0.2), (3, 0.2), (4, 0.2), (5, 0.04), (6,0.06), (7,0), (8,0)],
                1: [(0,0.3), (1, 0.04), (2,0.4), (3, 0.1), (4, 0.06), (5, 0.04), (6,0.06), (7,0), (8,0)]},
            8: {0: [(0, 0.09), (1, 0.19), (2, 0.21), (3, 0.2), (4, 0.21), (5, 0.05), (6, 0.05), (7, 0), (8, 0)],
                1: [(0, 0.29), (1, 0.05), (2, 0.41), (3, 0.1), (4, 0.05), (5, 0.05), (6, 0.05), (7, 0), (8, 0)]}
        }
        self.transitions = self.change_transition()

        # Initial state is A (state 0)
        self.state = 0
        self.R = np.zeros(len(self.observation_space))
        self.R[3] = 1

    def check_transitions(self):
        transitions = self.transitions
        for state, actions in transitions.items():
            for action, trans in actions.items():
                prob_sum = sum(prob for _, prob in trans)
                if abs(prob_sum - 1.0) > 1e-6:
                    print(f"⚠️ State {state}, Action {action} has total probability {prob_sum:.4f}")
                else:
                    print(f"✅ State {state}, Action {action} total probability sums to 1.")

    def change_transition(self):
        transitions = self.transitions
        # 初始化一个 [9, 2, 9] 的零矩阵
        P = np.zeros((9, 2, 9))

        # 遍历 transitions 字典，把对应的概率填进去
        for s in range(9):
            for a in range(2):
                for next_s, prob in transitions[s][a]:
                    P[s, a, next_s] = prob

        return P

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Choose the next state based on the transition probabilities
        #transitions = self.transitions[self.state][action]
        #next_state, _ = transitions[np.random.choice(len(transitions), p=[prob for _, prob in transitions])]

        probabilities = self.transitions[self.state, action]
        next_state = np.random.choice(len(probabilities), p=probabilities)

        # Define a reward function
        if next_state == 3:
            reward = 0.1
            done = True
        else:
            reward = 0  # Adjust as needed, for example based on states or transitions
            done = False

        return next_state, reward, done, {}

# Q-learning智能体
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])

    def get_q_table(self):
        return self.Q

# 训练主程序
def train_q_learning(episodes=100):
    env =  SimpleMDP()
    agent = QLearningAgent(n_states=len(env.observation_space), n_actions=2)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

    return agent
# Create the environment
agent = train_q_learning(episodes=1)
np.save("simple_MDP_Q_values.npy", agent.Q)
print("Final Q-Table:")
print(agent.get_q_table())









