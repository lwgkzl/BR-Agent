import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from tianshou.policy import A2CPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.discrete import Critic
from tianshou.utils.net.common import Net
import pickle
# from atari import create_atari_environment, preprocess_fn
import sys
sys.path.append('../')
from multiprocessing import Process, cpu_count, Lock
import os
import time
from Environment.MaskEnvrionment import MedicalEnvrionment
from NiceBayesian.ActorNet import MyActor
from a2c.Collect import MyCollector
from a2c.A2C import MyA2CPolicy
from a2c.Policy import Myonpolicy_trainer
# import sys
from torch import nn
# sys.path.append("./BayesNetwork")
import copy
from NiceBayesian.GradBayesv import GradBayesModel
from dataset.DataReader import get_dataset_information, load_disease_sym_pk
from pgmpy.models import BayesianModel
# logging.basicConfig(level=logging.INFO, filename='./log/test.txt', filemode='w')
import time
from a2c.Utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--step-per-epoch', type=int, default=16)
    parser.add_argument('--collect-per-step', type=int, default=64)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)

    # parser.add_argument(
    #     '--device', type=str,
    #     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str,
        default='cpu')

    # a2c special
    parser.add_argument('--vf-coef', type=float, default=1.0)  # 0.5 -- 0.9  0.1
    parser.add_argument('--ent-coef', type=float, default=0.001)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--max_episode_steps', type=int, default=22)
    parser.add_argument('--logpath', type=str, default='log/')
    parser.add_argument('--time_name', type=str, default=None)

    # my setting
    parser.add_argument('--transfor_grad', type=bool, default=False)
    parser.add_argument('--feature_model', type=int, default=4)
    parser.add_argument('--var_model', type=int, default=0)
    parser.add_argument('--random_rate', type=float, default=1.0)
    parser.add_argument('--max_turn', type=int, default=22)
    parser.add_argument('--prob_threshold', type=float, default=0.95)
    parser.add_argument('--save_model',type=bool, default=False)
    return parser.parse_args()


def save_fn(policy, save_model='./model/dxy/'):
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    # torch.save(policy.state_dict(), os.path.join(save_model, 'policy.pth'))
    torch.save(policy, os.path.join(save_model, 'policy_norm.pth'))

def test_a2c(args=None):
    slot_set = []
    with open('../dataset/dxy/dx_slot_set.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            slot_set.append(line.strip())
    args.max_turn = 22
    goals, total_disease_dict, total_sym_dict, bayes_data, bayes_graph, test_samples, test_ans = \
        get_dataset_information(data_path='../dataset/dxy/dxy_addnlu_goals.pk', disease_path='../dataset/dxy/dx_disease.txt',
                                symptom_path='../dataset/dxy/dx_symptom.txt')

    mutual_information = load_disease_sym_pk('../dataset/dxy/dx_mutual_information_norm.pk')
    relation_matrix = load_disease_sym_pk('../dataset/dxy/dx_relation_norm.pk')
    print("mutual_information: ", mutual_information.shape)
    print("relation_matrix: ", relation_matrix.shape)

    env = MedicalEnvrionment(slot_set, goals['test'])
    args.state_shape = env.observation_space.shape or env.observation_space.n

    args.action_shape = env.action_space.shape or env.action_space.n

    train_envs = SubprocVectorEnv(
        [lambda: MedicalEnvrionment(slot_set, goals['train'], max_turn=args.max_episode_steps, flag='train', disease_num=len(total_disease_dict))
         for _ in range(args.training_num)])

    test_envs = SubprocVectorEnv(
        [lambda: MedicalEnvrionment(slot_set, goals['test'], max_turn=args.max_episode_steps, flag="test", disease_num=len(total_disease_dict))
         for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    random.seed(args.seed)

    print("transfor_grad", args.transfor_grad)

    bayes_net = GradBayesModel(bayes_graph=bayes_graph, train_sample_data=bayes_data, disease_num=len(total_disease_dict), test_samples=test_samples, test_ans=test_ans)

    net = Net(args.layer_num, args.state_shape, device=args.device)

    actor = MyActor(args, bayes_net, args.action_shape, transformer_matrix=relation_matrix, transformer_matrix2=mutual_information).to(args.device)
    critic = Critic(net).to(args.device)

    conv5_params = list(map(id, actor.preprocess.parameters()))
    base_params = filter(lambda p: id(p) not in conv5_params,
                         actor.parameters())

    optim = torch.optim.Adam(
    [
        {'params': list(base_params)+list(critic.parameters())},
        {'params': list(actor.preprocess.parameters()), 'lr': args.lr2}
    ]
    , lr=args.lr)

    dist = torch.distributions.Categorical
    policy = MyA2CPolicy(
        actor, critic, optim, dist, args.gamma, vf_coef=args.vf_coef,
        ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm)
    # collector
    train_collector = MyCollector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = MyCollector(policy, test_envs)
    # log
    path = '#'.join(
        ["feature",
         "with_cof_" + str(args.vf_coef), 'lr:'+str(args.lr)])
    if args.time_name is None:
        time_name = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        # time_name = path
    else:
        time_name = args.time_name
    writer = SummaryWriter(os.path.join(args.logdir, args.logpath+time_name))

    result = Myonpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        len(goals['test']), args.batch_size, writer=writer, verbose=True, test_probs=False)
    path = path + '    ' + str(result['best_rate'])+"   mate_num_"+str(result['best_mate_num']) + "   best_len_"+str(result['best_len'])
    return result



if __name__ == '__main__':
    args1 = get_args()
    result = test_a2c(args1)