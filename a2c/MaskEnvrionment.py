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
        # 0表示未询问， -1 表示没有  1 表示有, 2表示已经问过了
        self.observation_space = spaces.Box(low=-1, high=2, shape=(len(self.slot_dict), ))
        self.state = None
        self.turn = 0
        self.goal = None
        self.action_mask = np.zeros(len(self.slot_dict))
        self.flag = flag
        self.disease_num = disease_num
        self.goal_num = 0
        self.mate_num = 0
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
        if self.flag=="test":
            print("sedd： ",seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.manual_seed(seed)
        return None

    def reset(self):
        self.turn = 0
        self.mate_num = 0
        if self.flag == 'train':
            self.goal = random.choice(self.start_set)
            self.goal_num = 0
        else:
            # print(self.goal_num)
            if self.goal_num >= len(self.start_set):
                self.goal_num = 0
            self.goal = self.start_set[self.goal_num]
            self.goal_num += 1
        self.state = np.zeros(len(self.slot_dict))
        positive_list, negative_list = self.trans_self_report(self.goal['explicit_inform_slots'])
        im_positive_list, im_negative_list = self.trans_self_report(self.goal['implicit_inform_slots'])

        self.state[positive_list] = [1] * len(positive_list)
        self.state[negative_list] = [-1] * len(negative_list)
        if self.flag == 'train':
            self.action_mask = np.ones(len(self.slot_dict))   # 初始化是1, 表示全都给mask掉
            self.action_mask[im_positive_list+im_negative_list] = [0] * len(im_positive_list+im_negative_list)  # 把需要预测的留下
            self.action_mask[:self.disease_num] = [0] * self.disease_num
        else:
            self.action_mask = np.zeros(len(self.slot_dict))
            self.action_mask[positive_list+negative_list] = [1] * len(positive_list+negative_list)
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action_name = self.num2slot[action]
        if self.goal['implicit_inform_slots'].get(action_name, False):
            self.state[action] = 1
            if int(action) >= self.disease_num and self.action_mask[int(action)] == 0:
                self.mate_num += 1
        elif action_name in self.goal['implicit_inform_slots'].keys():
            self.state[action] = -1
            if int(action) >= self.disease_num and self.action_mask[int(action)] == 0:
                self.mate_num += 1
        else:
            self.state[action] = 0
        self.action_mask[int(action)] = 1
        self.turn += 1
        # done = (np.abs(x) + np.abs(y) <= 1) or (np.abs(x) + np.abs(y) >= 2 * self.L + 1)
        done = (self.turn > self.max_turn) or (int(action) < self.disease_num)
        done = bool(done)
        is_right = False
        if not done:
            reward = -1
        else:
            if action_name == self.goal['disease_tag']:
                reward = 44
                is_right = True
            else:
                reward = -22
        return self.state, reward, done, {"history": self.action_mask, "right": is_right, "turn": self.turn, 'ans': action, 'mate_num': self.mate_num,'done': done}

    def render(self, mode='human'):
        return None

    def close(self):
        return None
