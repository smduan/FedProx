import torch
from fedavg.datasets import get_dataset, VRDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
import torch.nn.functional as F

class Server(object):

    def __init__(self, conf, model, test_df):

        self.conf = conf

        self.global_model = model

        self.test_dataset = get_dataset(conf, test_df)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=conf["batch_size"],shuffle=False)

    def model_aggregate(self, clients_model, weights):

        new_model = {}

        for name, params in self.global_model.state_dict().items():
            new_model[name] = torch.zeros_like(params)

        for key in clients_model.keys():

            for name, param in clients_model[key].items():
                new_model[name]= new_model[name] + clients_model[key][name] * weights[key]

        self.global_model.load_state_dict(new_model)

    @torch.no_grad()
    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        predict_prob = []
        labels = []
        
        predict = []

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.NLLLoss()
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.global_model(data)

            total_loss += criterion(output, target) # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            predict_prob.extend(F.softmax(output, dim=1).data[:,1].tolist())
            predict.extend(pred.data.cpu().tolist())
            labels.extend(target.data.cpu().tolist())
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()


        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size
        # roc = roc_auc_score(labels,predict_prob)
        roc = 0
        # f1 = f1_score(labels, predict)
        f1 = 0
        print("roc_auc = {0}, f1_score={1}, acc={2}".format(roc, f1, acc))

        return acc, total_l, roc, f1
#     @torch.no_grad()
#     def model_eval(self):
#         self.global_model.eval()                                   

#         total_loss = 0.0
#         correct = 0
#         dataset_size = 0
#         predict_prob = []
#         labels = []
#         predict = []
#         # criterion = torch.nn.CrossEntropyLoss()
#         criterion = torch.nn.NLLLoss()
#         for batch_id, batch in enumerate(self.test_loader):
#             data, target = batch
#             dataset_size += data.size()[0]

#             if torch.cuda.is_available():
#                 data = data.cuda()
#                 target = target.cuda()

#             _, output = self.global_model(data)

#             total_loss += criterion(output, target) # sum up batch loss
#             pred = output.data.max(1)[1]  # get the index of the max log-probability

#             predict_prob.extend(output.data[:,1].tolist())
#             labels.extend(target.data.cpu().tolist())
#             correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()


#         acc = 100.0 * (float(correct) / float(dataset_size))
#         total_l = total_loss.cpu().detach().numpy() / dataset_size
        
#         predict_binary = []
#         for item in predict_prob:
#             if item >=0:
#                 predict_binary.append(1)
#             else:
#                 predict_binary.append(0)
        
# #         print("labels:",labels)
# #         print("predict:",predict_prob)
# #         print("predict+:",predict_binary)
#         roc = roc_auc_score(labels,predict_prob)
        
#         f1 = f1_score(labels,predict_binary)
        
#         print("roc_auc = {}".format(roc_auc_score(labels,predict_prob)))
#         print("F1-Score = {:.4f}".format(f1_score(labels,predict_binary)))
#         return acc, total_l, roc, f1

    @torch.no_grad()
    def model_eval_vr(self, eval_vr, label):
        """
        :param eval_vr:
        :param label:
        :return: ?????????????????????
        """

        self.retrain_model.eval()

        eval_dataset = VRDataset(eval_vr, label)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.functional.cross_entropy()
        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.retrain_model(data)

            total_loss += criterion(output, target)  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size
        return acc, total_l

    def retrain_vr(self, vr, label, eval_vr, classifier):
        """
        :param vr:
        :param label:
        :return: ???????????????????????????
        """
        self.retrain_model = classifier
        retrain_dataset = VRDataset(vr, label)
        retrain_loader = torch.utils.data.DataLoader(retrain_dataset, batch_size=self.conf["batch_size"],shuffle=True)

        optimizer = torch.optim.SGD(self.retrain_model.parameters(), lr=self.conf['retrain']['lr'], momentum=self.conf['momentum'],weight_decay=self.conf["weight_decay"])
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(self.conf["retrain"]["epoch"]):

            self.retrain_model.train()

            for batch_id, batch in enumerate(retrain_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.retrain_model(data)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()

            acc, eval_loss = self.model_eval_vr(eval_vr, label)
            print("Retraining epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, loss, eval_loss, acc))

        return self.retrain_model

    def cal_global_gd(self,client_mean, client_cov, client_length):
        """
        :param client_mean: ??????????????????????????????, ??????
        :param client_cov:  ??????????????????????????????????????????
        :param client_length: ???????????????????????????????????? ??????
        :return:
        """

        g_mean = []
        g_cov = []

        clients = list(client_mean.keys())

        for c in range(len(client_mean[clients[0]])):

            mean_c = np.zeros_like(client_mean[clients[0]][0])
            n_c = 0
            # ??????c???????????????
            for k in clients:
                n_c += client_length[k][c]

            cov_ck = np.zeros_like(client_cov[clients[0]][0])
            mul_mean = np.zeros_like(client_cov[clients[0]][0])

            for k in clients:
                # ??????c????????????????????????
                mean_c += (client_length[k][c] / n_c) * np.array(client_mean[k][c])  # ??????(3)

                mean_ck = np.array(client_mean[k][c])
                mul_mean += ((client_length[k][c]) / (n_c - 1)) * np.dot(mean_ck.T, mean_ck)

                cov_ck += ((client_length[k][c] - 1) / (n_c - 1)) * np.array(client_cov[k][c])

            g_mean.append(mean_c)
            cov_c = cov_ck + mul_mean - (n_c / (n_c - 1)) * np.dot(mean_c.T, mean_c)  ##?????????4???

            g_cov.append(cov_c)

        return g_mean, g_cov


    def get_feature_label(self):
        self.global_model.eval()
        
        cnt = 0
        features = []
        true_labels = []
        pred_labels = []
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            cnt += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            feature, output = self.global_model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            
            features.append(feature)
            true_labels.append(target)
            pred_labels.append(pred)
            
            if cnt > 1000:
                break

        features = torch.cat(features, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)

        return features, true_labels, pred_labels
