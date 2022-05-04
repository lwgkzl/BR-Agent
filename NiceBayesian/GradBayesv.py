#!/usr/bin/env python3

import itertools
from collections import defaultdict
import logging
import numpy as np
from NiceBayesian.CPD import TabularCPD
import torch
from torch import nn
from pgmpy.models import BayesianModel
from NiceBayesian.ExactInference import VariableElimination
from NiceBayesian.BayesianModel import MYBayesianModel
# from NiceBayesian.untils import logging
import random
# 不更新，或者随机初始化  flag=0 表示一切正常， flag=1表示不更新参数，  flag=2 表示随机初始化,但是贝叶斯的参数更新， flag=3表示随机初始化，但是贝叶斯的参数表不更新
class GradBayesModel(nn.Module):
    """
    Base class for Bayesian Models.
    """

    def __init__(self, bayes_graph=None, train_sample_data=None, disease_num=4, flag=0, chose_index=[], test_samples=None, test_ans=None):
        super(GradBayesModel, self).__init__()
        bayes = BayesianModel(bayes_graph)
        bayes.fit(train_sample_data)
        xx = bayes.get_cpds()
        self.flag = flag
        new_cpd = []
        para_list = []
        if self.flag == 2 or self.flag == 3:
            print(f"random_bayes with random_rate {len(chose_index)/len(xx)}")
        num = 0
        for ind, cpd in enumerate(xx):
            para_value = nn.Parameter(torch.tensor(cpd.get_values()[0]), requires_grad=True)
            if self.flag == 2 or self.flag == 3:
                if ind in chose_index:
                    para_value = nn.Parameter(torch.rand(para_value.size()), requires_grad=True)
                    num += 1
            para_list.append(para_value)
        self.cpd_list = nn.ParameterList(para_list)  # TODO 只有一半的变量是参数，另一半是1-p就好
        for k, cpd in enumerate(xx):
            evidence = cpd.variables[1:] if len(cpd.variables) > 1 else None
            evidence_card = cpd.cardinality[1:] if len(cpd.variables) > 1 else None

            value = torch.ones_like(self.cpd_list[k]) - self.cpd_list[k]
            value = torch.cat((self.cpd_list[k].unsqueeze(0), value.unsqueeze(0)), 0)

            n_cpd = TabularCPD(cpd.variable, 2, value, evidence, evidence_card, cpd.state_names.copy())
            new_cpd.append(n_cpd)
        self.new_model = MYBayesianModel(bayes_graph)
        self.new_model.add_cpds(new_cpd)
        self.minn = 0.0001
        self.infer = VariableElimination(self.new_model)
        self.disease_num = disease_num
        self.first_test_turn = True
        self.test_evidece = test_samples
        self.test_ans = test_ans


    def forward(self, test_sample_data):  # forward的输入应该是个tensor
        if not self.training:
            self.first_test_turn = True

        test_sample_data = test_sample_data.long()
        bs = test_sample_data.size(0)

        if self.flag != 1 and self.flag != 3 and self.training:
            for k, para in enumerate(self.cpd_list):
                self.cpd_list[k].data = torch.clamp(para.data, self.minn, 1.0 - self.minn)
                value = torch.ones_like(self.cpd_list[k]) - self.cpd_list[k]
                value = torch.cat((self.cpd_list[k].unsqueeze(0), value.unsqueeze(0)), 0)
                self.new_model.cpds[k].values = value

            self.infer = VariableElimination(self.new_model)
        if self.training and self.first_test_turn:
            self.first_test_turn = False
            num = 0

            for k, evid in enumerate(self.test_evidece):
                total_key = [k for k in evid.keys()]
                for key in total_key:
                    if key not in self.new_model.nodes:
                        evid.pop(key)
                q = self.infer.query(variables=[str(i) for i in range(self.disease_num)], evidence=evid, joint=False, show_progress=False)
                ans_list = []
                for item in q.values():
                    ans_list.append(item.values[1])
                x = np.argmax(np.array(ans_list))
                if x == self.test_ans[k]:
                    num += 1
            print("static test: ", num / len(self.test_evidece))


        ans = torch.ones((bs, self.disease_num))
        for i in range(bs):
            one_evidence = {}
            for k, v in enumerate(test_sample_data[i]):
                if str(k) not in self.new_model.nodes():
                    continue
                if v != 0 and v != 2:
                    one_evidence[str(k)] = v

            q = self.infer.query(variables=[str(i) for i in range(self.disease_num)], evidence=one_evidence, joint=False,show_progress=False)
            for x, item in enumerate(q.values()):
                ans[i][x] = item.values[1]  # 预测为1的概率是多少
        return ans
