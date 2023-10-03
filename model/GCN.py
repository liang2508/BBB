# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score,confusion_matrix,cohen_kappa_score,matthews_corrcoef
from sklearn.metrics import roc_curve, auc
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
import sys

sys.path.append(r'D:\\Research Project\\cydmaster\\BBB\\model\\random_split\\ablation_study')
from model import EncodeSmileBatch, GCN, GCN_none

os.chdir('D:\\Research Project\\cydmaster\\BBB\\model\\random_split\\ablation_study')

def CalculateMetrics(y_true,y_score):
    fpr,tpr,threshold = roc_curve(y_true, y_score) ###计算真正率和假正率
    roc_auc = round(auc(fpr,tpr),3) ###计算auc的值
    y_score[y_score>0.5] = 1
    y_score[y_score<=0.5] = 0
    acc = round(accuracy_score(y_true,y_score),3)
    precision = round(precision_score(y_true,y_score),3)
    recall = round(recall_score(y_true,y_score),3)
    f1 = round(f1_score(y_true,y_score),3)
    kappa = round(cohen_kappa_score(y_true,y_score),3)
    mcc = round(matthews_corrcoef(y_true,y_score),3)
    cm = confusion_matrix(y_true,y_score) ###cm[1,1]第一个1代表真实值为1 第二个1代表预测值为1
    sensitivity = round(cm[1,1]/(cm[1,1]+cm[1,0]),3)
    specificity = round(cm[0,0]/(cm[0,0]+cm[0,1]),3)
    BAC = round((sensitivity+specificity)/2,3)
    data = np.array([roc_auc,BAC,f1,mcc,kappa,acc,precision,recall,sensitivity,specificity])
    data = data.reshape(1, -1)
    return data

def training(epoch,x_train,smi,y_train,train_batches=10,batch_size=250):
    model.train()
    train_loss = 0
    for batch_idx in range(train_batches):
        optimizer.zero_grad()
        data = x_train[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_train))].to(device)
        label = y_train[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_train))].to(device)
        smi_batch = smi[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_train))]
        adj,feature,total_atoms = EncodeSmileBatch(smi_batch)
        weights = label.clone().view(-1)
        weights[weights==0] = weight
        outputs = model(adj,feature,data)
        outputs = torch.sigmoid(outputs)
        loss = F.binary_cross_entropy(outputs.view(-1),label.view(-1),weight=weights)
        train_loss += loss.item()
        loss.backward()   
        optimizer.step()
    return train_loss / train_batches

def testing(epoch,x_test,smi,y_test,test_batches=10,batch_size=250):
    model.eval()
    test_loss = 0
    y_true_list=np.array([])
    y_score_list=np.array([])
    for batch_idx in range(test_batches):
        data = x_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))].to(device)
        label = y_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))].to(device)
        smi_batch = smi[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(x_test))]
        adj,feature,total_atoms = EncodeSmileBatch(smi_batch)
        weights=label.clone().view(-1)
        weights[weights==0]=weight
        outputs= model(adj,feature,data)
        outputs=torch.sigmoid(outputs)
        loss = F.binary_cross_entropy(outputs.view(-1),label.view(-1),weight=weights)
        test_loss += loss.item()
        y_true_list=np.append(y_true_list,label.long().detach().cpu().numpy())
        y_score_list=np.append(y_score_list,outputs.view(-1).detach().cpu().numpy())
    return test_loss / test_batches,y_true_list,y_score_list

def training_none(epoch,smi,y_train,train_batches=10,batch_size=250):
    model.train()
    train_loss = 0
    for batch_idx in range(train_batches):
        optimizer.zero_grad()
        label = y_train[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(smi))].to(device)
        smi_batch = smi[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(smi))]
        adj,feature,total_atoms = EncodeSmileBatch(smi_batch)
        weights = label.clone().view(-1)
        weights[weights==0] = weight
        outputs = model(adj,feature)
        outputs = torch.sigmoid(outputs)
        loss = F.binary_cross_entropy(outputs.view(-1),label.view(-1),weight=weights)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / train_batches

def testing_none(epoch,smi,y_test,test_batches=10,batch_size=250):
    model.eval()
    test_loss = 0
    y_true_list=np.array([])
    y_score_list=np.array([])
    for batch_idx in range(test_batches):
        label = y_test[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(smi))].to(device)
        smi_batch = smi[batch_idx*batch_size:min((batch_idx+1)*batch_size,len(smi))]
        adj,feature,total_atoms = EncodeSmileBatch(smi_batch)
        weights=label.clone().view(-1)
        weights[weights==0] = weight
        outputs= model(adj,feature)
        outputs=torch.sigmoid(outputs)
        loss = F.binary_cross_entropy(outputs.view(-1),label.view(-1),weight=weights)
        test_loss += loss.item()
        y_true_list=np.append(y_true_list,label.long().detach().cpu().numpy())
        y_score_list=np.append(y_score_list,outputs.view(-1).detach().cpu().numpy())
    return test_loss / test_batches,y_true_list,y_score_list
'''
########## none ##########
dataset = './data_split/'
for i in range(1,11):
    print('none start:', i)
    y_train = list(pd.read_csv(dataset + 'all_train_data_feature_selection_{0}.csv'.format(i)).label)
    y_train = np.array(y_train)
    y_test = list(pd.read_csv(dataset + 'all_test_data_feature_selection_{0}.csv'.format(i)).label)
    y_test = np.array(y_test)
    weight = y_train.sum()/(y_train.shape[0]-y_train.sum())
    x_train_smi = list(pd.read_csv(dataset + 'all_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_test_smi = list(pd.read_csv(dataset + 'all_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    # 定义GCN参数
    max_atom = 50
    edge_dim = 4
    atom_type_dim = 14
    in_channels = 14
    out_channels = 32
    linear1 = 128
    linear2 = 64
    linear3 = 32
    linear4 = 10
    dropout = 0.5
    ###########训练模型#############
    torch.backends.cudnn.enabled = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 128
    if len(x_train_smi) % batch_size == 0:
        train_batches = int(len(x_train_smi) / batch_size)
    else:
        train_batches = int(len(x_train_smi) / batch_size) + 1

    if len(x_test_smi) % batch_size == 0:
        test_batches = int(len(x_test_smi) / batch_size)
    else:
        test_batches = int(len(x_test_smi) / batch_size) + 1
    torch.cuda.empty_cache()
    model = GCN_none(max_atom,edge_dim,atom_type_dim,in_channels,out_channels,linear1,linear2,linear3,linear4,dropout).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss = training_none(epoch, x_train_smi, y_train, train_batches=train_batches, batch_size=batch_size)
        test_loss, _, _ = testing_none(epoch, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
     #   print ('epoch: {}, {:.2f}, {:.2f}'.format(epoch,train_loss,test_loss))
    _, y_true_test, y_score_test = testing_none(epoch, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
    if i == 1:
        statistics = CalculateMetrics(y_true_test, y_score_test)
    else:
        statistics_test = CalculateMetrics(y_true_test, y_score_test)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    torch.save(model, './result/GCN_none_{0}.pkl'.format(i))

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/GCN_statistics_none.csv',header=True)


########## Rdkit2d ##########
dataset = './data_split/'
for i in range(1,11):
    print('rdkit2d start:', i)
    data_std = pd.read_csv(dataset + 'rdkit2d_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    scaler = preprocessing.StandardScaler().fit(data_std)
    #读取数据
    X_train = pd.read_csv(dataset + 'rdkit2d_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_train = np.array(X_train)
    X_train = scaler.transform(X_train)
    y_train = list(pd.read_csv(dataset + 'rdkit2d_train_data_feature_selection_{0}.csv'.format(i)).label)
    y_train = np.array(y_train)
    X_test = pd.read_csv(dataset + 'rdkit2d_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = list(pd.read_csv(dataset + 'rdkit2d_test_data_feature_selection_{0}.csv'.format(i)).label)
    y_test = np.array(y_test)
    weight = y_train.sum()/(y_train.shape[0]-y_train.sum())
    x_train_smi = list(pd.read_csv(dataset + 'rdkit2d_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_test_smi = list(pd.read_csv(dataset + 'rdkit2d_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_train = torch.tensor(np.array(X_train)).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(np.array(X_test)).float()
    y_test = torch.tensor(y_test).float()
    # 定义GCN参数
    max_atom = 50
    edge_dim = 4
    atom_type_dim = 14
    in_channels = 14
    out_channels = 32
    property_dim = X_test.shape[1]
    linear1 = 128
    linear2 = 64
    linear3 = 32
    linear4 = 10
    dropout = 0.5
    ###########训练模型#############
    torch.backends.cudnn.enabled = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 128
    if len(x_train) % batch_size == 0:
        train_batches = int(len(x_train) / batch_size)
    else:
        train_batches = int(len(x_train) / batch_size) + 1

    if len(x_test) % batch_size == 0:
        test_batches = int(len(x_test) / batch_size)
    else:
        test_batches = int(len(x_test) / batch_size) + 1
    torch.cuda.empty_cache()
    model = GCN(max_atom,edge_dim,atom_type_dim,in_channels,out_channels,property_dim,linear1,linear2,linear3,linear4,dropout).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss = training(epoch, x_train, x_train_smi, y_train, train_batches=train_batches, batch_size=batch_size)
        test_loss, _, _ = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
     #   print ('epoch: {}, {:.2f}, {:.2f}'.format(epoch,train_loss,test_loss))
    _, y_true_test, y_score_test = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
    if i == 1:
        statistics = CalculateMetrics(y_true_test, y_score_test)
    else:
        statistics_test = CalculateMetrics(y_true_test, y_score_test)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    torch.save(model, './result/GCN_rdkit2d_{0}.pkl'.format(i))

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/GCN_statistics_rdkit2d.csv',header=True)


########## ecfp4 ##########
dataset = './data_split/'
for i in range(1,11):
    print('ecfp4 start:', i)
    data_std = pd.read_csv(dataset + 'ecfp4_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    scaler = preprocessing.StandardScaler().fit(data_std)
    #读取数据
    X_train = pd.read_csv(dataset + 'ecfp4_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_train = np.array(X_train)
    X_train = scaler.transform(X_train)
    y_train = list(pd.read_csv(dataset + 'ecfp4_train_data_feature_selection_{0}.csv'.format(i)).label)
    y_train = np.array(y_train)
    X_test = pd.read_csv(dataset + 'ecfp4_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = list(pd.read_csv(dataset + 'ecfp4_test_data_feature_selection_{0}.csv'.format(i)).label)
    y_test = np.array(y_test)
    weight = y_train.sum()/(y_train.shape[0]-y_train.sum())
    x_train_smi = list(pd.read_csv(dataset + 'ecfp4_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_test_smi = list(pd.read_csv(dataset + 'ecfp4_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_train = torch.tensor(np.array(X_train)).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(np.array(X_test)).float()
    y_test = torch.tensor(y_test).float()
    # 定义GCN参数
    max_atom = 50
    edge_dim = 4
    atom_type_dim = 14
    in_channels = 14
    out_channels = 32
    property_dim = X_test.shape[1]
    linear1 = 128
    linear2 = 64
    linear3 = 32
    linear4 = 10
    dropout = 0.5
    ###########训练模型#############
    torch.backends.cudnn.enabled = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 128
    if len(x_train) % batch_size == 0:
        train_batches = int(len(x_train) / batch_size)
    else:
        train_batches = int(len(x_train) / batch_size) + 1

    if len(x_test) % batch_size == 0:
        test_batches = int(len(x_test) / batch_size)
    else:
        test_batches = int(len(x_test) / batch_size) + 1
    torch.cuda.empty_cache()
    model = GCN(max_atom,edge_dim,atom_type_dim,in_channels,out_channels,property_dim,linear1,linear2,linear3,linear4,dropout).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss = training(epoch, x_train, x_train_smi, y_train, train_batches=train_batches, batch_size=batch_size)
        test_loss, _, _ = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
     #   print ('epoch: {}, {:.2f}, {:.2f}'.format(epoch,train_loss,test_loss))
    _, y_true_test, y_score_test = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
    if i == 1:
        statistics = CalculateMetrics(y_true_test, y_score_test)
    else:
        statistics_test = CalculateMetrics(y_true_test, y_score_test)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    torch.save(model, './result/GCN_ecfp4_{0}.pkl'.format(i))

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/GCN_statistics_ecfp4.csv',header=True)


########## Rdkit3d ##########
dataset = './data_split/'
for i in range(1,11):
    print('rdkit3d start:', i)
    data_std = pd.read_csv(dataset + 'rdkit3d_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    scaler = preprocessing.StandardScaler().fit(data_std)
    #读取数据
    X_train = pd.read_csv(dataset + 'rdkit3d_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_train = np.array(X_train)
    X_train = scaler.transform(X_train)
    y_train = list(pd.read_csv(dataset + 'rdkit3d_train_data_feature_selection_{0}.csv'.format(i)).label)
    y_train = np.array(y_train)
    X_test = pd.read_csv(dataset + 'rdkit3d_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = list(pd.read_csv(dataset + 'rdkit3d_test_data_feature_selection_{0}.csv'.format(i)).label)
    y_test = np.array(y_test)
    weight = y_train.sum()/(y_train.shape[0]-y_train.sum())
    x_train_smi = list(pd.read_csv(dataset + 'rdkit3d_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_test_smi = list(pd.read_csv(dataset + 'rdkit3d_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_train = torch.tensor(np.array(X_train)).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(np.array(X_test)).float()
    y_test = torch.tensor(y_test).float()
    # 定义GCN参数
    max_atom = 50
    edge_dim = 4
    atom_type_dim = 14
    in_channels = 14
    out_channels = 32
    property_dim = X_test.shape[1]
    linear1 = 128
    linear2 = 64
    linear3 = 32
    linear4 = 10
    dropout = 0.5
    ###########训练模型#############
    torch.backends.cudnn.enabled = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 128
    if len(x_train) % batch_size == 0:
        train_batches = int(len(x_train) / batch_size)
    else:
        train_batches = int(len(x_train) / batch_size) + 1

    if len(x_test) % batch_size == 0:
        test_batches = int(len(x_test) / batch_size)
    else:
        test_batches = int(len(x_test) / batch_size) + 1
    torch.cuda.empty_cache()
    model = GCN(max_atom,edge_dim,atom_type_dim,in_channels,out_channels,property_dim,linear1,linear2,linear3,linear4,dropout).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss = training(epoch, x_train, x_train_smi, y_train, train_batches=train_batches, batch_size=batch_size)
        test_loss, _, _ = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
     #   print ('epoch: {}, {:.2f}, {:.2f}'.format(epoch,train_loss,test_loss))
    _, y_true_test, y_score_test = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
    if i == 1:
        statistics = CalculateMetrics(y_true_test, y_score_test)
    else:
        statistics_test = CalculateMetrics(y_true_test, y_score_test)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    torch.save(model, './result/GCN_rdkit3d_{0}.pkl'.format(i))

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/GCN_statistics_rdkit3d.csv',header=True)
'''

########## all ##########
dataset = './data_split/'
for i in range(1,11):
    print('all start:', i)
    data_std = pd.read_csv(dataset + 'all_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    scaler = preprocessing.StandardScaler().fit(data_std)
    #读取数据
    X_train = pd.read_csv(dataset + 'all_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_train = np.array(X_train)
    X_train = scaler.transform(X_train)
    y_train = list(pd.read_csv(dataset + 'all_train_data_feature_selection_{0}.csv'.format(i)).label)
    y_train = np.array(y_train)
    X_test = pd.read_csv(dataset + 'all_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,2:]
    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)
    y_test = list(pd.read_csv(dataset + 'all_test_data_feature_selection_{0}.csv'.format(i)).label)
    y_test = np.array(y_test)
    weight = y_train.sum()/(y_train.shape[0]-y_train.sum())
    x_train_smi = list(pd.read_csv(dataset + 'all_train_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_test_smi = list(pd.read_csv(dataset + 'all_test_data_feature_selection_{0}.csv'.format(i)).iloc[:,0])
    x_train = torch.tensor(np.array(X_train)).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(np.array(X_test)).float()
    y_test = torch.tensor(y_test).float()
    # 定义GCN参数
    max_atom = 50
    edge_dim = 4
    atom_type_dim = 14
    in_channels = 14
    out_channels = 32
    property_dim = X_test.shape[1]
    linear1 = 128
    linear2 = 64
    linear3 = 32
    linear4 = 10
    dropout = 0.5
    ###########训练模型#############
    torch.backends.cudnn.enabled = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 128
    if len(x_train) % batch_size == 0:
        train_batches = int(len(x_train) / batch_size)
    else:
        train_batches = int(len(x_train) / batch_size) + 1

    if len(x_test) % batch_size == 0:
        test_batches = int(len(x_test) / batch_size)
    else:
        test_batches = int(len(x_test) / batch_size) + 1
    torch.cuda.empty_cache()
    model = GCN(max_atom,edge_dim,atom_type_dim,in_channels,out_channels,property_dim,linear1,linear2,linear3,linear4,dropout).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss = training(epoch, x_train, x_train_smi, y_train, train_batches=train_batches, batch_size=batch_size)
        test_loss, _, _ = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
     #   print ('epoch: {}, {:.2f}, {:.2f}'.format(epoch,train_loss,test_loss))
    _, y_true_test, y_score_test = testing(epoch, x_test, x_test_smi, y_test, test_batches=test_batches, batch_size=batch_size)
    if i == 1:
        statistics = CalculateMetrics(y_true_test, y_score_test)
    else:
        statistics_test = CalculateMetrics(y_true_test, y_score_test)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    torch.save(model, './result/GCN_all_{0}.pkl'.format(i))

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/GCN_statistics_all.csv',header=True)