import pickle
import json
import random
import numpy as np
import torch
import copy
import pandas as pd
seed = 1626
np.random.seed(seed)
random.seed(seed)

def read_file(file_path):
    total_item = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            total_item.append(line.strip())
    return total_item


def get_disease_and_symptom(data_path, disease_path, symptom_path):
    with open(data_path, 'rb') as f:  # goal_dict_ori
        goal = pickle.load(f)

    total_disease = read_file(disease_path)
    total_sym = read_file(symptom_path)

    total_disease_dict = dict(zip(total_disease, [i for i in range(len(total_disease))]))
    disease_num = len(total_disease_dict)

    total_sym_dict = dict(zip(total_sym, [i+disease_num for i in range(len(total_sym))]))
    symptom_num = len(total_sym_dict)

    return goal, total_disease_dict, total_sym_dict, disease_num, symptom_num


def load_disease_sym_txt(path):
    line_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_list.append(np.array(line.split(), dtype=float))
    return np.array(line_list)


def load_disease_sym_pk(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def make_samples(goals_set, total_disease_dict, total_sym_dict):
    symptom_num = len(total_sym_dict)
    disease_num = len(total_disease_dict)
    test_samples = np.full((len(goals_set), symptom_num + disease_num), -1)
    test_ans = []
    keys_list = []
    for i, dialog in enumerate(goals_set):
        disease = dialog['disease_tag']
        disease_index = total_disease_dict[disease]
        test_samples[i][disease_index] = 1
        #         disease_index[i]
        test_ans.append(disease_index)
        for k, v in dialog['implicit_inform_slots'].items():
            v_index = total_sym_dict[k]
            if v:
                test_samples[i][v_index] = 1
            else:
                test_samples[i][v_index] = -1
        for k, v in dialog['explicit_inform_slots'].items():
            v_index = total_sym_dict[k]
            if v:
                test_samples[i][v_index] = 1
            else:
                test_samples[i][v_index] = -1
    return test_samples, test_ans

def make_test_data(goals_set, total_disease_dict, total_sym_dict):
    evidence_list = []
    test_ans = []
    for i,dialog in enumerate(goals_set):
        one_evidence = {}
        disease = dialog['disease_tag']
        disease_index =  total_disease_dict[disease]
        test_ans.append(disease_index)
        for k, v in dialog['implicit_inform_slots'].items():
            v_index = total_sym_dict[k]
            if v:
                one_evidence[str(v_index)] = 1
            else:
                one_evidence[str(v_index)] = -1
        for k, v in dialog['explicit_inform_slots'].items():
            v_index = total_sym_dict[k]
            if v:
                one_evidence[str(v_index)] = 1
            else:
                one_evidence[str(v_index)] = -1
        evidence_list.append(one_evidence)
    return evidence_list, test_ans

def get_bayes_struct(goal, total_disease_dict, total_sym_dict, threshold=5):
    # disease_sym 存储每一个疾病与其他哪些症状有关系  sym_disease存储每一个症状与哪些疾病有关系
    # disease_num = len(total_disease_dict)
    sym_disease = {}
    disease_sym = {}
    # yan = np.zeros(disease_num, symptom_num)
    disease_sym_num = {}
    for dialog in goal['train']:
        disease = dialog['disease_tag']
        true_set = set()
        for k, v in dialog['implicit_inform_slots'].items():
            if v:
                true_set.add(k)
                #             yan[total_disease_dict[disease]][total_sym_dict[k]]+=1
                if k not in sym_disease.keys():
                    sym_disease[k] = set()
                sym_disease[k].add(disease)
        for k, v in dialog['explicit_inform_slots'].items():
            if v:
                true_set.add(k)
                #             yan[total_disease_dict[disease]][total_sym_dict[k]]+=1
                if k not in sym_disease.keys():
                    sym_disease[k] = set()
                sym_disease[k].add(disease)
        if disease not in disease_sym.keys():
            disease_sym[disease] = set()
            disease_sym_num[disease] = {}
        for s in true_set:
            disease_sym_num[disease][s] = disease_sym_num[disease].get(s, 0) + 1
        disease_sym[disease].update(true_set)
    bayes_graph = []
    for k, v in disease_sym_num.items():
        disease = total_disease_dict[k]
        for kk, vv in disease_sym_num[k].items():
            # print(kk,vv)
            if vv >= threshold:
                bayes_graph.append(tuple([str(disease), str(total_sym_dict[kk])]))
    # print("bayes edges num: ", len(bayes_graph))  # 边的条数
    return bayes_graph

def get_dataset_information(data_path, disease_path, symptom_path, random_rate=1.0, edge_threthod=5):
    goal, total_disease_dict, total_sym_dict, disease_num, symptom_num = \
        get_disease_and_symptom(data_path, disease_path, symptom_path)   # 需要保留结构，只减少数据
    train_len = len(goal['train'])
    choice_index = random.sample(list(np.arange(train_len)), int(train_len * random_rate))
    train_samples, train_ans = make_samples([goal['train'][i] for i in choice_index], total_disease_dict, total_sym_dict)
    test_samples, test_ans = make_test_data(goal['test'], total_disease_dict, total_sym_dict)
    bayes_data = pd.DataFrame(train_samples, columns=[str(i) for i in range(disease_num + symptom_num)])
    # train_sample_data = pd.DataFrame(train_samples[:, disease_num:], columns=[str(i + disease_num) for i in range(symptom_num)])
    # test_sample_data = pd.DataFrame(test_samples[:, disease_num:], columns=[str(i + disease_num) for i in range(symptom_num)])

    bayes_graph = get_bayes_struct(goal, total_disease_dict, total_sym_dict, edge_threthod)
    return goal, total_disease_dict, total_sym_dict, bayes_data, bayes_graph, test_samples, test_ans