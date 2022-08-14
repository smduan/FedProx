import json,os

import pandas as pd
import torch
import copy
import numpy as np
import random

from fedavg.server import Server
from fedavg.client import Client,Client_fedprox
from fedavg.models import CNN_Model,weights_init_normal, ReTrainModel,MLP
from utils import get_data
from conf import conf

from collections import Counter
from sklearn.preprocessing import MinMaxScaler

mu = 0.05

def min_max_norm(train_datasets, test_dataset, cat_columns, label):
    
    train_data = None
    for key in train_datasets.keys():
        train_datasets[key]['tag'] = key
        train_data = pd.concat([train_data, train_datasets[key]])
    test_dataset['tag'] = key+1
    data = pd.concat([train_data, test_dataset])
    
    min_max = MinMaxScaler()
    con = []

    #查找连续列
    for c in data.columns:
        #TODO 这里写死了使用clinical，后续再修改
        if c not in cat_columns and c not in [label, 'tag']:
            con.append(c)

    data[con] = min_max.fit_transform(data[con])

    # 离散列one-hot
    data = pd.get_dummies(data, columns=cat_columns)
    
    for key in train_datasets.keys():
        c_data = data[data['tag'] == key]
        c_data = c_data.drop(columns=['tag'])
        train_datasets[key] = c_data
    
    test_dataset = data[data['tag'] == key+1]
    test_dataset = test_dataset.drop(columns=['tag'])

    return train_datasets, test_dataset

def test_model(model, test_df):
    """
    model: state_dict
    test_df: test dataset dataframe
    """
    test_dataset = MyImageDataset(test_df,conf["data_column"], conf["label_column"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf["batch_size"],shuffle=False,drop_last = False)
    
#     test_model = VGG('VGG11').cuda()
    test_model = CNN_Model().cuda()
    test_model.load_state_dict(model)
    
    test_model.eval()
    
    total_loss = 0.0
    correct = 0
    dataset_size = 0

    criterion = torch.nn.CrossEntropyLoss()
    for batch_id, batch in enumerate(test_loader):
        data, target = batch
        dataset_size += data.size()[0]

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        _, output = test_model(data)

        total_loss += criterion(output, target) # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss.cpu().detach().numpy() / dataset_size
    return acc, total_l

def model_init(conf, train_datasets, test_dataset, device):
    ###初始化每个节点聚合权值
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)
    print("各节点的聚合权值为：", client_weight)
    
    clients = {}
    
    ##训练目标模型
    if conf['model_name'] == "mlp":
        n_input = test_dataset.shape[1] - 1
        model = MLP(n_input, 512, conf["num_classes"])
    elif conf['model_name'] == 'cnn':
        model = CNN_Model()
    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda(device=device)
    
    server = Server(conf, model, test_dataset)
    print("Server初始化完成!")
    
    for key in train_datasets.keys():
        clients[key] = Client_fedprox(conf, copy.deepcopy(server.global_model), train_datasets[key])#new_add
    print("参与方初始化完成！")

    #保存模型
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    
    return clients, server, client_weight

def train_and_eval(clients, server, client_weight):
    #联邦训练
    loss_list = []
    acc_list = []
    roc_list = []
    max_auc = 0
    max_acc = 0
    maxe = 0
    maxe2 = 0
    for e in range(conf["global_epochs"]):
        global_weight_collector = list((server.global_model).parameters())#new_add
        clients_models = {}
        for key in clients.keys():
#             print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model,global_weight_collector,mu)#new_add
            clients_models[key] = copy.deepcopy(model_k)

    #         acc, loss = test_model(clients_models[key], test_dataset)
    #         print("client %d,Epoch %d, global_acc: %f, global_loss: %f\n" % (key, e, acc, loss))


        #联邦聚合
        server.model_aggregate(clients_models, client_weight)

        #测试全局模型
        acc, loss, auc_roc, f1 = server.model_eval()
        loss_list.append(loss)
        acc_list.append(acc)
        roc_list.append(auc_roc)
#         auc_roc, loss = server.model_eval()
    #     print("Epoch %d, global_acc: %f, global_loss: %f, auc_roc: %f, f1: %f\n" % (e, acc, loss, auc_roc, f1))
        print("Epoch %d, global_loss: %f, auc_roc: %f" % (e, loss, auc_roc))

        #保存最好的模型
        if auc_roc > max_auc:
            torch.save(server.global_model.state_dict(), 
                       os.path.join(conf["model_dir"], "roc_model_best_{}.pth".format(e)))
            
            for idx,_ in enumerate(clients_models):
                torch.save(clients_models[idx], 
                           os.path.join(conf["model_dir"], "roc_model_best_l{}_{}.pth".format(idx,e)))
#                 print(clients_models[idx])
#                 print("roc")
                exit()
                idx += 1
            
#             print("model save done !")
            max_auc = auc_roc
            maxe = e
        if acc > max_acc:
            torch.save(server.global_model.state_dict(), 
                       os.path.join(conf["model_dir"], "acc_model_best.pth"))
#             print("model save done !")
            
            for idx, _ in enumerate(clients_models):
                torch.save(clients_models[idx], 
                           os.path.join(conf["model_dir"], "acc_model_best_l{}.pth".format(idx)))
#                 print(clients_models[idx])
#                 print("acc")
#                 exit()
            max_acc = acc
            maxe2 = e

    print('max auc = {0}, epoch = {1}'.format(max_auc, maxe))
    print('max acc = {0}, epoch = {1}'.format(max_acc, maxe2))
#     loss_list.to_csv(os.path.join(conf["model_dir"], "loss_list.csv"),index=False)
#     acc_list.to_csv(os.path.join(conf["model_dir"], "acc_list.csv"),index=False)
#     roc_list.to_csv(os.path.join(conf["model_dir"], "roc_list.csv"),index=False)
    save = [loss_list,acc_list,roc_list]
    name = ["loss","acc","roc"]
    record = pd.DataFrame(data=save)
    record.to_csv(os.path.join(conf["model_dir"], "record_list.csv"),index=False)
    return max_auc

def base_train(conf, dataset_name, b, clients_num, path, label_name, train_gpu):
    # 构造文件读取路径
    train_files_path_list = [path + "beta{}/".format(b) + dataset_name + "_node_{}.csv".format(i) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    
    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, train_gpu)
    
    max_score = train_and_eval(clients, server, client_weight)
    
    return max_score

def base_train_60(conf, dataset_name, b, clients_num, path, label_name, train_gpu):
    
    # 构造文件读取路径
    data_idx_list=random.sample(range(0,60),clients_num)
    train_files_path_list = [path + "beta{}/".format(b) + dataset_name + "_node_{}.csv".format(data_idx_list[i]) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    
    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, train_gpu)
    
    max_score = train_and_eval(clients, server, client_weight)
    
    return max_score

def get_norm_test(conf, dataset_name, b, clients_num, path, label_name, train_gpu):
    # 构造文件读取路径
    train_files_path_list = [path + "beta{}/".format(b) + dataset_name + "_node_{}.csv".format(i) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    test_dataset.to_csv(path + "beta{}/".format(b) + dataset_name + "_norm_test.csv",index=False)
    
    return 

def get_random_data(syn_data, aug_numbers,label,ratio):
    """
    从合成数据中采样增强
    """
  
    aug_data = None   
    for i in range(len(aug_numbers)):
        
        aug_i = syn_data[syn_data[label] == i]
        if aug_i.shape[0] >= aug_numbers[i]:
            aug_data = pd.concat([aug_data, aug_i.sample(int(ratio * aug_numbers[i]))])
        else:
            print('label {} has no enough synthetic data'.format(i))
    
    return aug_data
        
def random_aug(train_datasets, path, dataset_name, label, label_num,  aug_type='same_number'):
    """
    随机增强
    """ 
    labels_dis = []
    
    for key in train_datasets.keys():
        label_dis = []
        for i in range(label_num):
            label_i = len(train_datasets[key][train_datasets[key][label] == i])
            label_dis.append(label_i)
        labels_dis.append(label_dis)
    labels_dis = np.array(labels_dis)
    print(labels_dis)
    total_dis = np.sum(labels_dis, axis=0)
    print(total_dis)
    aug_numbers = total_dis - labels_dis
    print("len:",train_datasets.keys())
    ratio = 1/len(train_datasets.keys())
    if aug_type == 'same_number':
        
        for key in train_datasets.keys():
#             syn_data = pd.read_csv('./data/clinical/syn_data/clinical_syn_{}.csv'.format(key))
            syn_data = pd.read_csv('{0}/{1}_syn.csv'.format(path, dataset_name, key))
            aug_data = get_random_data(syn_data, aug_numbers[key],label,ratio)
            train_datasets[key] = pd.concat([train_datasets[key], aug_data])
            print(train_datasets[key].shape)
    
    return train_datasets

def random_aug_from_one(train_datasets, path, dataset_name, label, label_num,  aug_type='same_number'):
    """
    随机增强
    """ 
    labels_dis = []
    
    for key in train_datasets.keys():
        label_dis = []
        for i in range(label_num):
            label_i = len(train_datasets[key][train_datasets[key][label] == i])
            label_dis.append(label_i)
        labels_dis.append(label_dis)
    labels_dis = np.array(labels_dis)
    print(labels_dis)
    total_dis = np.sum(labels_dis, axis=0)
    print(total_dis)
    aug_numbers = total_dis - labels_dis

    if aug_type == 'same_number':
        
        for key in train_datasets.keys():
#             syn_data = pd.read_csv('./data/clinical/syn_data/clinical_syn_{}.csv'.format(key))
            syn_data = pd.read_csv('{0}/{1}_syn.csv'.format(path, dataset_name, key))
            aug_data = get_random_data(syn_data, aug_numbers[key],label)
            train_datasets[key] = pd.concat([train_datasets[key], aug_data])
            print(train_datasets[key].shape)
    
    return train_datasets

def augment_train(conf, dataset_name, b, clients_num, path, label_name, label_num, augment_path, train_gpu):
    # 构造文件读取路径
    train_files_path_list = [path + "beta{}/".format(b) + dataset_name + "_node_{}.csv".format(i) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets = random_aug(train_datasets, augment_path, dataset_name, label_name, label_num)
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    
    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, train_gpu)
    
    max_score = train_and_eval(clients, server, client_weight)
    
    return max_score
    #
def get_norm_augment_test(conf, dataset_name, b, clients_num, path, label_name, label_num, augment_path, train_gpu):
    # 构造文件读取路径
    train_files_path_list = [path + "beta{}/".format(b) + dataset_name + "_node_{}.csv".format(i) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets = random_aug(train_datasets, augment_path, dataset_name, label_name, label_num)
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    
    test_dataset.to_csv(path + "beta{}/".format(b) + dataset_name + "_norm_augment_test.csv",index=False)
    return

def augment_train_in_60(conf, dataset_name, b, clients_num, path, label_name, label_num, augment_path, train_gpu):
    # 构造文件读取路径
    data_idx_list=random.sample(range(0,60),clients_num)
    train_files_path_list = [path + "beta{}/".format(b) + dataset_name + "_node_{}.csv".format(data_idx_list[i]) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets = random_aug(train_datasets, augment_path, dataset_name, label_name, label_num)
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    
    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, train_gpu)
    
    max_score = train_and_eval(clients, server, client_weight)
    
    return max_score

def gan_augment_train(conf, dataset_name, b, clients_num, path, label_name, label_num, augment_path, train_gpu):
    # 构造文件读取路径
    train_files_path_list = [path + "b={}/".format(b) + label_name + "_{}.csv".format(i) for i in range(clients_num)]
    print("划分数据目录如下:\n" + str(train_files_path_list))
    
    # 读取文件
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("测试数据格式如下: " + str(test_dataset.shape))
    
    train_datasets = random_aug_from_one(train_datasets, augment_path, dataset_name, label_name, label_num)
    
    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset, 
        conf['discrete_columns'][dataset_name], 
        conf['label_column']) 
    
    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, train_gpu)
    
    max_score = train_and_eval(clients, server, client_weight)
    
    return max_score



