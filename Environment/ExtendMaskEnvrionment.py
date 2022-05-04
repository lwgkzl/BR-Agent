import gym
from gym import spaces
import numpy as np
import random
from collections import defaultdict
# 做数据增强

def extend_dialog(dialog, expect_len=-1):
    new_dialog = {'disease_tag': dialog['disease_tag']}
    total_key = [k for k in dialog['implicit_inform_slots'].keys()] +[k for k in dialog['explicit_inform_slots'].keys()]
    totak_dict = dialog['implicit_inform_slots'].copy()
    totak_dict.update(dialog['explicit_inform_slots'].copy())
    ex_len = len(dialog['explicit_inform_slots'])
    if expect_len >= len(total_key) or expect_len==-1:
        choice_ken = random.sample(total_key, ex_len)
    else:
        choice_ken = random.sample(total_key, expect_len)
    new_dialog['explicit_inform_slots'] = {k:totak_dict[k] for k in choice_ken}
    new_dialog['implicit_inform_slots'] = {k:totak_dict[k] for k in list(set(total_key)-set(choice_ken))}
    return new_dialog

def Mean_data(goal):
    total_data = []
    for dialog in goal:
        total_data.append(dialog)
        ran_key = random.randint(1,3)
        total_data.append(extend_dialog(dialog, ran_key))
    return total_data



class MedicalEnvrionment(gym.Env):

    # 把贝叶斯网络直接当成actor，  适合NN
    def __init__(self, slot_set, start_set, max_turn=22, flag='train', disease_num=4):
        self.max_turn = max_turn
        # self.slot_set = slot_set
        self.slot_dict = {v: k for k, v in enumerate(slot_set)}
        self.num2slot = {v: k for k, v in self.slot_dict.items()}
        if flag == 'train':
            self.start_set = Mean_data(start_set)
        else:
            self.start_set = start_set
        self.action_space = spaces.Discrete(len(self.slot_dict))
        # print("action_spa")
        # print("action_space: ", self.action_space)
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
        # print(self.flag, action)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action_name = self.num2slot[action]
        if self.goal['implicit_inform_slots'].get(action_name, False):
            self.state[action] = 1
            self.mate_num += 1
        elif action_name in self.goal['implicit_inform_slots'].keys():
            self.state[action] = -1
            self.mate_num += 1
        else:
            self.state[action] = 2
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
        return self.state, reward, done, {"history": self.action_mask, "right": is_right, "turn": self.turn, 'ans': action, 'mate_num': self.mate_num}

    def render(self, mode='human'):
        return None

    def close(self):
        return None