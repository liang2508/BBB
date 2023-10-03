
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer,precision_score, recall_score,f1_score,accuracy_score,confusion_matrix,cohen_kappa_score,matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from scipy import interp
from itertools import cycle
import pickle

os.chdir('D:\\Research Project\\cydmaster\\BBB\\model\\random_split\\ablation_study')

def CalculateMetrics(y_true,y_pred,y_score):
    fpr,tpr,threshold = roc_curve(y_true, y_score) ###计算真正率和假正率
    roc_auc = round(auc(fpr,tpr),3) ###计算auc的值
    acc = round(accuracy_score(y_true,y_pred),3)
    precision = round(precision_score(y_true,y_pred),3)
    recall = round(recall_score(y_true,y_pred),3)
    f1 = round(f1_score(y_true,y_pred),3)
    kappa = round(cohen_kappa_score(y_true,y_pred),3)
    mcc = round(matthews_corrcoef(y_true,y_pred),3)
    cm = confusion_matrix(y_true,y_pred)###cm[1,1]第一个1代表真实值为1 第二个1代表预测值为1
    sensitivity = round(cm[1,1]/(cm[1,1]+cm[1,0]),3)
    specificity = round(cm[0,0]/(cm[0,0]+cm[0,1]),3)
    BAC = round((sensitivity+specificity)/2,3)
    data = np.array([roc_auc,BAC,f1,mcc,kappa,acc,precision,recall,sensitivity,specificity])
    data = data.reshape(1,-1)
    return data

'''
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
    #先搜索最优参数
    params = {'n_estimators': range(10, 760, 50), 'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}
    estimator = GridSearchCV(XGBClassifier(scale_pos_weight=1 / weight, seed=0),
                             params, cv=5, refit=True, n_jobs=12, scoring=make_scorer(cohen_kappa_score))
    estimator.fit(X_train, y_train)
    best_param = estimator.best_params_
    clf = XGBClassifier(n_estimators=best_param['n_estimators'],learning_rate=best_param['learning_rate'],subsample=0.96,colsample_bytree=0.91,scale_pos_weight=1/weight,seed=0)
    clf.fit(X_train,y_train)
    train_predicted = clf.predict(X_train)
    test_predicted = clf.predict(X_test)
    test_score = clf.predict_proba(X_test)[:,1]   #取预测为1即正例的概率
    if i == 1:
        statistics = CalculateMetrics(y_test,test_predicted,test_score)
    else:
        statistics_test = CalculateMetrics(y_test,test_predicted,test_score)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    ##输出模型
    with open('./result/XGB_rdkit2d_{0}.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f)
        f.close()

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/XGB_statistics_rdkit2d.csv',header=True)


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
    #先搜索最优参数
    params = {'n_estimators': range(10, 760, 50), 'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}
    estimator = GridSearchCV(XGBClassifier(scale_pos_weight=1 / weight, seed=0),
                             params, cv=5, refit=True, n_jobs=12, scoring=make_scorer(cohen_kappa_score))
    estimator.fit(X_train, y_train)
    best_param = estimator.best_params_
    clf = XGBClassifier(n_estimators=best_param['n_estimators'],learning_rate=best_param['learning_rate'],subsample=0.96,colsample_bytree=0.91,scale_pos_weight=1/weight,seed=0)
    clf.fit(X_train,y_train)
    train_predicted = clf.predict(X_train)
    test_predicted = clf.predict(X_test)
    test_score = clf.predict_proba(X_test)[:,1]   #取预测为1即正例的概率
    if i == 1:
        statistics = CalculateMetrics(y_test,test_predicted,test_score)
    else:
        statistics_test = CalculateMetrics(y_test,test_predicted,test_score)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    ##输出模型
    with open('./result/XGB_rdkit3d_{0}.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f)
        f.close()

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/XGB_statistics_rdkit3d.csv',header=True)


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
    #先搜索最优参数
    params = {'n_estimators': range(10, 760, 50), 'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}
    estimator = GridSearchCV(XGBClassifier(scale_pos_weight=1 / weight, seed=0),
                             params, cv=5, refit=True, n_jobs=12, scoring=make_scorer(cohen_kappa_score))
    estimator.fit(X_train, y_train)
    best_param = estimator.best_params_
    clf = XGBClassifier(n_estimators=best_param['n_estimators'],learning_rate=best_param['learning_rate'],subsample=0.96,colsample_bytree=0.91,scale_pos_weight=1/weight,seed=0)
    clf.fit(X_train,y_train)
    train_predicted = clf.predict(X_train)
    test_predicted = clf.predict(X_test)
    test_score = clf.predict_proba(X_test)[:,1]   #取预测为1即正例的概率
    if i == 1:
        statistics = CalculateMetrics(y_test,test_predicted,test_score)
    else:
        statistics_test = CalculateMetrics(y_test,test_predicted,test_score)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    ##输出模型
    with open('./result/XGB_ecfp4_{0}.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f)
        f.close()

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/XGB_statistics_ecfp4.csv',header=True)
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
    #先搜索最优参数
    params = {'n_estimators': range(10, 760, 50), 'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}
    estimator = GridSearchCV(XGBClassifier(scale_pos_weight=1 / weight, seed=0),
                             params, cv=5, refit=True, n_jobs=10, scoring=make_scorer(cohen_kappa_score))
    estimator.fit(X_train, y_train)
    best_param = estimator.best_params_
    clf = XGBClassifier(n_estimators=best_param['n_estimators'],learning_rate=best_param['learning_rate'],subsample=0.96,colsample_bytree=0.91,scale_pos_weight=1/weight,seed=0)
    clf.fit(X_train,y_train)
    train_predicted = clf.predict(X_train)
    test_predicted = clf.predict(X_test)
    test_score = clf.predict_proba(X_test)[:,1]   #取预测为1即正例的概率
    if i == 1:
        statistics = CalculateMetrics(y_test,test_predicted,test_score)
    else:
        statistics_test = CalculateMetrics(y_test,test_predicted,test_score)
        statistics = pd.DataFrame(np.vstack((statistics, statistics_test)))
    ##输出模型
    with open('./result/XGB_all_{0}.pkl'.format(i), 'wb') as f:
        pickle.dump(clf, f)
        f.close()

statistics.columns = ['AUC','BAC','F1','MCC','kappa','acc','precision','recall','sensitivity','specificity']
statistics.index=['Test1','Test2','Test3','Test4','Test5','Test6','Test7','Test8','Test9','Test10']
statistics.to_csv('./result/XGB_statistics_all.csv',header=True)
