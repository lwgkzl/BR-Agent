import pickle
from dataset.DataReader import get_dataset_information, load_disease_sym_pk
goals, total_disease_dict, total_sym_dict, bayes_data, bayes_graph, test_samples, test_ans = \
    get_dataset_information(data_path='../dataset/dxy/dxy_addnlu_goals.pk',
                            disease_path='../dataset/dxy/dx_disease.txt',
                            symptom_path='../dataset/dxy/dx_symptom.txt')

print("dxy: ", len(bayes_graph))

with open('ehr/ehr_15_with_dev.pk','rb') as f:
    ehr_data = pickle.load(f)
print('ehr: ', len(ehr_data))

with open('muzhi/bayes_graph_5_14.pk','rb') as f:
    muzhi_data = pickle.load(f)
print('muzhi: ', len(muzhi_data))