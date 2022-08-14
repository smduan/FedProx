import json,os
from traceback import print_tb

from conf import conf
import pandas as pd
import torch
import numpy as np
from fedavg.server import Server
from fedavg.client import Client,Client_fedprox

from fedavg.models import CNN_Model,weights_init_normal, ReTrainModel,MLP
from utils import get_data
import copy
from sklearn.preprocessing import MinMaxScaler


def min_max_norm(train_datasets, test_dataset, cat_columns, label):
    
    train_data = None
    for key in train_datasets.keys():

        train_datasets[key]['tag'] = key

        #########################
        # print('===========================')
        
        # print(key)
        # print(train_datasets[key]['tag'])
        # print(train_datasets[key])

        #########################

        train_data = pd.concat([train_data, train_datasets[key]])
    print("train's shape:")
    print(train_data.shape)
    test_dataset['tag'] = key+1
    data = pd.concat([train_data, test_dataset])
    # print(data)
    print("here:concat data's shape:")
    print(data.shape)
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


if __name__ == '__main__':
    
    mu = 0.05

    # cat_columns = ["anaemia","diabetes","high_blood_pressure","sex","smoking"]
#     cat_columns = ['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1',
#  'Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6',
#  'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11',
#  'Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16',
#  'Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21',
#  'Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26',
#  'Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31',
#  'Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36',
#  'Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']
    cat_columns = ['protocol_type', 'service', 'flag']

    if conf['use_data'] == 'random':
        train_datasets, test_dataset = get_data()
        i = 0
        while os.path.exists(f"./data/clinical/tmp/beta{conf['beta']}_{i}/"):
            i += 1
        os.makedirs(f"./data/clinical/tmp/beta{conf['beta']}_{i}")
        for key in train_datasets.keys():
            print("======================")
            print(key,"_shape:")
            print(train_datasets[key].shape)
            train_datasets[key].to_csv(f"./data/clinical/tmp/beta{conf['beta']}_{i}/clinical_node_{key}.csv", index=False)

    elif conf['use_data'] == 'stored':
        # train_files = [f"./data/clinical/tmp/beta{conf['beta']}/original_data/clinical_node_0.csv",
        #                f"./data/clinical/tmp/beta{conf['beta']}/original_data/clinical_node_1.csv",
        #                f"./data/clinical/tmp/beta{conf['beta']}/original_data/clinical_node_2.csv",
        #                f"./data/clinical/tmp/beta{conf['beta']}/original_data/clinical_node_3.csv",
        #                f"./data/clinical/tmp/beta{conf['beta']}/original_data/clinical_node_4.csv"]
        # train_files = [f"./data/covtype/tmp/beta{conf['beta']}/covtype_node_0.csv",
        #                f"./data/covtype/tmp/beta{conf['beta']}/covtype_node_1.csv",
        #                f"./data/covtype/tmp/beta{conf['beta']}/covtype_node_2.csv",
        #                f"./data/covtype/tmp/beta{conf['beta']}/covtype_node_3.csv",
        #                f"./data/covtype/tmp/beta{conf['beta']}/covtype_node_4.csv"]
        train_files = [f"./data/intrusion/tmp/beta{conf['beta']}/intrusion_node_0.csv",
                       f"./data/intrusion/tmp/beta{conf['beta']}/intrusion_node_1.csv",
                       f"./data/intrusion/tmp/beta{conf['beta']}/intrusion_node_2.csv",
                       f"./data/intrusion/tmp/beta{conf['beta']}/intrusion_node_3.csv",
                       f"./data/intrusion/tmp/beta{conf['beta']}/intrusion_node_4.csv"]
        train_datasets = {}

        for i in range(len(train_files)):
            train_datasets[i] = pd.read_csv(train_files[i])
            # print(train_datasets[i]['label'].value_counts())

        # test_dataset = pd.read_csv('./data/clinical/clinical_test.csv')
        # test_dataset = pd.read_csv('./data/covtype/covtype_test.csv')
        test_dataset = pd.read_csv('./data/intrusion/intrusion_test.csv')
        print("test_data's shape:")
        print(test_dataset.shape)
    
    else:
        raise ValueError

    train_datasets, test_dataset = min_max_norm(train_datasets, test_dataset, cat_columns, 'label') 
    print(test_dataset.shape)
    for key in train_datasets.keys():
        print("======================")
        print(key,"_shape:")
        print(train_datasets[key].shape)
        # train_datasets[key].to_csv(f"./data/clinical/tmp/beta{conf['beta']}/norm_data/clinical_node_{key}.csv", index=False)
    print("norm_test's shape:")
    print(test_dataset.shape)
    
    # test_dataset.to_csv('./data/clinical/norm_clinical_test.csv', index=False)

    ###初始化每个节点聚合权值
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("聚合权值初始化")
    print(client_weight)

    ##保存节点
    clients = {}
    # 保存节点模型
    clients_models = {}

    if conf['model_name'] == "mlp":
        n_input = test_dataset.shape[1] - 1
        model = MLP(n_input, 512, conf["num_classes"])
    elif conf['model_name'] == 'cnn':
        ##训练目标模型
        model = CNN_Model()
    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda()

    server = Server(conf, model, test_dataset)

    print("Server初始化完成!")

    for key in train_datasets.keys():
        clients[key] = Client_fedprox(conf, copy.deepcopy(server.global_model), train_datasets[key])

    print("参与方初始化完成！")

    # 保存模型
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    max_auc = 0
    
    #联邦训练
    for e in range(conf["global_epochs"]):
        
        global_weight_collector = list((server.global_model).parameters())
        
        for key in clients.keys():
            print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model,global_weight_collector,mu)
            clients_models[key] = copy.deepcopy(model_k)
            
    #         acc, loss = test_model(clients_models[key], test_dataset)
    #         print("client %d,Epoch %d, global_acc: %f, global_loss: %f\n" % (key, e, acc, loss))
        
        
        #联邦聚合
        server.model_aggregate(clients_models, client_weight)
        
        #测试全局模型
        acc, loss, auc_roc, f1 = server.model_eval()
        print("Epoch %d, global_acc: %f, global_loss: %f, auc_roc: %f, f1: %f\n" % (e, acc, loss, auc_roc, f1))
        
        #保存最好的模型
        if  auc_roc >= max_auc:
            torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], "model-epoch{}.pth".format(e)))
            print("model save done !")
            max_auc = auc_roc
            maxe = e
            
    print('max auc = {0}, epoch = {1}'.format(max_auc, maxe))
