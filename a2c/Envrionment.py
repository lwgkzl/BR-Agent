import gym
from gym import spaces
import numpy as np
import random

class MedicalEnvrionment(gym.Env):

    def __init__(self, slot_set, start_set, max_turn=22, flag='train', disease_num=4):
        self.max_turn = max_turn
        # self.slot_set = slot_set
        self.slot_dict = {v: k for k, v in enumerate(slot_set)}
        self.num2slot = {v: k for k, v in self.slot_dict.items()}
        self.start_set = start_set
        self.action_space = spaces.Discrete(len(self.slot_dict))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.slot_dict), ))
        self.state = None
        self.turn = 0
        self.goal = None
        self.action_mask = np.zeros(len(self.slot_dict))
        self.flag = flag
        self.goal_num = -1
        self.disease_num = disease_num
        if self.flag == 'train':
            self.goal_num = 0

    def trans_self_report(self, report_dict):
        true_list, false_list = [], []
        for k, v in report_dict.items():
            if v:
                true_list.append(self.slot_dict[k])
            else:
                false_list.append(self.slot_dict[k])
        return true_list, false_list

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return None

    def reset(self):
        self.turn = 0
        if self.flag == 'train':
            self.goal = random.choice(self.start_set)
            self.goal_num = 0
        else:
            if self.goal_num >= len(self.start_set):
                self.goal_num = 0
                return self.state
            self.goal = self.start_set[self.goal_num]
            self.goal_num += 1
        self.state = np.zeros(len(self.slot_dict))
        positive_list, negative_list = self.trans_self_report(self.goal['explicit_inform_slots'])
        self.state[positive_list] = [1] * len(positive_list)
        self.state[negative_list] = [-1] * len(negative_list)
        self.action_mask = np.zeros(len(self.slot_dict))
        self.action_mask[positive_list+negative_list] = [1] * len(positive_list+negative_list)
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action_name = self.num2slot[action]
        if self.goal['implicit_inform_slots'].get(action_name, False):
            self.state[action] = 1
        elif action_name in self.goal['implicit_inform_slots'].keys():
            self.state[action] = -1
        else:
            self.state[action] = 2
        rere = -1
        self.action_mask[int(action)] = 1
        self.turn += 1
        done = (self.turn > self.max_turn) or (int(action) < self.disease_num)
        done = bool(done)
        is_right = False
        if not done:
            reward = rere
        else:
            if action_name == self.goal['disease_tag']:
                reward = 44
                is_right = True
            else:
                reward = -22

        return self.state, reward, done, {"history": self.action_mask, "right": is_right, "ans": action}

    def render(self, mode='human'):
        return None

    def close(self):
        return None

