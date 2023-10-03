
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors,Descriptors3D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import Standardizer
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import csv
import math
import os
os.chdir('D:\\Research Project\\cydmaster\\BBB\\model\\random_split\\ablation_study')

def smi_to_mol(smi):
    def mod():
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                 Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=False)
        return mol
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return mod()
    except:
        mol = mod()
    return mol

def randomize_smiles(smis,num):
    smis_new = []
    for smi in smis:
        mol = smi_to_mol(smi)
        for i in range(num):
            try:
                smi_new = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                mol = Chem.AddHs(Chem.MolFromSmiles(smi_new))
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                mol.GetConformer()
                smis_new.append(smi_new)
            except:
                print('error, {0}:{1}'.format(i, smi_new))
    smis_new = list(set(smis_new))
    return smis_new

def standardize_smiles(smis):
    smis_new = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)
        standardizer = Standardizer()
        mol = standardizer.charge_parent(mol)
        smis_new.append(Chem.MolToSmiles(mol))
    return smis_new

def calculate_property(smile,label):
    property_list, property_name = [], []
    property_name += ['smile', 'label']
    property_list += [smile, label]
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)  ###加氢
    AllChem.EmbedMolecule(m)
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except:
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)  ###加氢
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMolecule(m)
    # 计算ECFP_4
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024).ToBitString()
    fp = [int(i) for i in list(fp)]
    property_list += fp
    property_name += [i + 1 for i in range(1024)]
    # 计算RDKIT2D
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    feature_list = list(desc_calc.CalcDescriptors(m))
    # 'NumHDonors', 'NumRotatableBonds', 'MolLogP', 'TPSA', 'MolWt',
    property_list += feature_list
    property_name += descs
    # 计算RDKIT3D
    feature_list = [Descriptors3D.Asphericity(m),
                    Descriptors3D.Asphericity(m),
                    Descriptors3D.InertialShapeFactor(m),
                    Descriptors3D.NPR1(m),
                    Descriptors3D.NPR2(m),
                    Descriptors3D.PMI1(m),
                    Descriptors3D.PMI2(m),
                    Descriptors3D.PMI3(m),
                    Descriptors3D.RadiusOfGyration(m),
                    Descriptors3D.SpherocityIndex(m),
                    ]
    property_list += feature_list
    property_name += ['Asphericity', 'Asphericity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2',
                      'PMI3', 'RadiusOfGyration', 'SpherocityIndex']
    feature_list = rdMolDescriptors.CalcAUTOCORR3D(m)
    property_list += feature_list
    property_name += ['AUTOCORR3D_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcRDF(m)
    property_list += feature_list
    property_name += ['RDF_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcMORSE(m)
    property_list += feature_list
    property_name += ['MORSE_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcWHIM(m)
    property_list += feature_list
    property_name += ['WHIM_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcGETAWAY(m)
    property_list += feature_list
    property_name += ['GETAWAY_' + str(i + 1) for i in range(len(feature_list))]
    return property_list, property_name

def calculate_property_rdkit2d(smile,label):
    property_list, property_name = [], []
    property_name += ['smile', 'label']
    property_list += [smile, label]
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)  ###加氢
    # 计算RDKIT2D
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    feature_list = list(desc_calc.CalcDescriptors(m))
    # 'NumHDonors', 'NumRotatableBonds', 'MolLogP', 'TPSA', 'MolWt',
    property_list += feature_list
    property_name += descs
    return property_list, property_name

def calculate_property_rdkit3d(smile,label):
    property_list, property_name = [], []
    property_name += ['smile', 'label']
    property_list += [smile, label]
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)  ###加氢
    AllChem.EmbedMolecule(m)
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except:
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)  ###加氢
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMolecule(m)
    # 计算RDKIT3D
    feature_list = [Descriptors3D.Asphericity(m),
                    Descriptors3D.Asphericity(m),
                    Descriptors3D.InertialShapeFactor(m),
                    Descriptors3D.NPR1(m),
                    Descriptors3D.NPR2(m),
                    Descriptors3D.PMI1(m),
                    Descriptors3D.PMI2(m),
                    Descriptors3D.PMI3(m),
                    Descriptors3D.RadiusOfGyration(m),
                    Descriptors3D.SpherocityIndex(m),
                    ]
    property_list += feature_list
    property_name += ['Asphericity', 'Asphericity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2',
                      'PMI3', 'RadiusOfGyration', 'SpherocityIndex']
    feature_list = rdMolDescriptors.CalcAUTOCORR3D(m)
    property_list += feature_list
    property_name += ['AUTOCORR3D_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcRDF(m)
    property_list += feature_list
    property_name += ['RDF_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcMORSE(m)
    property_list += feature_list
    property_name += ['MORSE_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcWHIM(m)
    property_list += feature_list
    property_name += ['WHIM_' + str(i + 1) for i in range(len(feature_list))]
    feature_list = rdMolDescriptors.CalcGETAWAY(m)
    property_list += feature_list
    property_name += ['GETAWAY_' + str(i + 1) for i in range(len(feature_list))]
    return property_list, property_name

def calculate_property_ecfp4(smile,label):
    property_list, property_name = [], []
    property_name += ['smile', 'label']
    property_list += [smile, label]
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)  ###加氢
    # 计算ECFP_4
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024).ToBitString()
    fp = [int(i) for i in list(fp)]
    property_list += fp
    property_name += [i + 1 for i in range(1024)]
    return property_list, property_name

def calculate_properties(smiles,labels):
    property_lists = []
    _, property_name = calculate_property(smiles[0],labels[0])
    for i in tqdm(range(len(smiles))):
        try:
            property_list, property_name = calculate_property(smiles[i],labels[i])
            property_lists.append(property_list)
        except:
            continue
    return property_name, property_lists

def calculate_properties_rdkit2d(smiles,labels):
    property_lists = []
    _, property_name = calculate_property_rdkit2d(smiles[0],labels[0])
    for i in tqdm(range(len(smiles))):
        property_list, property_name = calculate_property_rdkit2d(smiles[i],labels[i])
        property_lists.append(property_list)
    return property_name, property_lists

def calculate_properties_rdkit3d(smiles,labels):
    property_lists = []
    _, property_name = calculate_property_rdkit3d(smiles[0],labels[0])
    for i in tqdm(range(len(smiles))):
        try:
            property_list, property_name = calculate_property_rdkit3d(smiles[i],labels[i])
            property_lists.append(property_list)
        except:
            continue
    return property_name, property_lists

def calculate_properties_ecfp4(smiles,labels):
    property_lists = []
    _, property_name = calculate_property_ecfp4(smiles[0],labels[0])
    for i in tqdm(range(len(smiles))):
        property_list, property_name = calculate_property_ecfp4(smiles[i],labels[i])
        property_lists.append(property_list)
    return property_name, property_lists

def shuffle(array1,array2):
    # random shuffle two arrays and keep corresponding elements the same
    permutation = np.random.permutation(len(array1))
    shuffled_array1 = array1[permutation]
    shuffled_array2 = array2[permutation]
    return shuffled_array1, shuffled_array2

def data_preprocess(smis, labels):
    # remove smis with atom nums of less than 5
    idx = []
    for i in tqdm(range(len(smis))):
        mol = Chem.MolFromSmiles(smis[i])
        if mol is None:
            idx.append(i)
            continue
        elif mol.GetNumAtoms() < 5 or mol.GetNumAtoms() > 50:
            idx.append(i)
            continue
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            mol.GetConformer()
        except:
            idx.append(i)
    smis_new = [smi for j, smi in enumerate(smis) if j not in idx]
    labels_new = [label for j, label in enumerate(labels) if j not in idx]
    # remove ions in salt molecule
    smis_new = standardize_smiles(smis_new)
    return smis_new, labels_new

def split(smis_new, labels_new, property):
    # split positive and negative
    idx_pos, idx_neg = [m for m, ele in enumerate(labels_new) if ele == 1], [m for m, ele in enumerate(labels_new) if ele == 0]
    smis_pos, smis_neg = [smi for j, smi in enumerate(smis_new) if j in idx_pos], [smi for j, smi in enumerate(smis_new) if j in idx_neg]
    labels_pos, labels_neg = [label for j, label in enumerate(labels_new) if j in idx_pos], [label for j, label in enumerate(labels_new) if j in idx_neg]
    # random split train and test of positive and negative
    smis_pos, labels_pos, smis_neg, labels_neg = np.array(smis_pos), np.array(labels_pos), np.array(smis_neg), np.array(labels_neg)
  #  smis_train_pos, smis_test_pos, labels_train_pos, labels_test_pos = train_test_split(smis_pos, labels_pos, test_size=0.2, random_state=1)
  #  smis_train_neg, smis_test_neg, labels_train_neg, labels_test_neg = train_test_split(smis_neg, labels_neg, test_size=0.2, random_state=1)
    smis_train_pos, smis_test_pos, labels_train_pos, labels_test_pos = train_test_split(smis_pos, labels_pos, test_size=0.2)
    smis_train_neg, smis_test_neg, labels_train_neg, labels_test_neg = train_test_split(smis_neg, labels_neg, test_size=0.2)
    # data augmentation
    smis_train_pos, smis_test_pos = randomize_smiles(smis_train_pos, 5), randomize_smiles(smis_test_pos, 5)
    smis_train_neg, smis_test_neg = randomize_smiles(smis_train_neg, 3), randomize_smiles(smis_test_neg, 3)
    labels_train_pos, labels_test_pos = [1] * len(smis_train_pos), [1] * len(smis_test_pos)
    labels_train_neg, labels_test_neg = [0] * len(smis_train_neg), [0] * len(smis_test_neg)
    print('train_pos:',len(smis_train_pos))
    print('test_pos:',len(smis_test_pos))
    print('train_neg:',len(smis_train_neg))
    print('test_neg:',len(smis_test_neg))
    # random shuffle two arrays
    smis_train, labels_train = np.concatenate((smis_train_pos, smis_train_neg)), np.concatenate(
        (labels_train_pos, labels_train_neg))
    smis_test, labels_test = np.concatenate((smis_test_pos, smis_test_neg)), np.concatenate(
        (labels_test_pos, labels_test_neg))
    smis_train_new, labels_train_new = shuffle(smis_train, labels_train)
    smis_test_new, labels_test_new = shuffle(smis_test, labels_test)
    # calculate descriptors
    if property == 'rdkit2d':
        property_name, properties_train = calculate_properties_rdkit2d(smis_train_new, labels_train_new)
        property_name, properties_test = calculate_properties_rdkit2d(smis_test_new, labels_test_new)
    elif property == 'rdkit3d':
        property_name, properties_train = calculate_properties_rdkit3d(smis_train_new, labels_train_new)
        property_name, properties_test = calculate_properties_rdkit3d(smis_test_new, labels_test_new)
    elif property == 'ecfp4':
        property_name, properties_train = calculate_properties_ecfp4(smis_train_new, labels_train_new)
        property_name, properties_test = calculate_properties_ecfp4(smis_test_new, labels_test_new)
    elif property == 'all':
        property_name, properties_train = calculate_properties(smis_train_new, labels_train_new)
        property_name, properties_test = calculate_properties(smis_test_new, labels_test_new)
    return property_name, properties_train, properties_test

def check_nan_v1(data1, data2):
    # 如果某个描述符有nan值，则用均值代替
    for column in range(data1.shape[1]):
        idx = []
        for row in range(data1.shape[0]):
            if math.isnan(data1.iloc[row, column]) or math.isinf(data1.iloc[row, column]):
                idx.append(row)
        values = [value for value in list(data1.iloc[:, column]) if not math.isnan(value)]
        values = [value for value in values if not math.isinf(value)]
        for i in idx:
            data1.iloc[i, column] = np.mean(values)
    for column in range(data2.shape[1]):
        idx = []
        for row in range(data2.shape[0]):
            if math.isnan(data2.iloc[row, column]) or math.isinf(data2.iloc[row, column]):
                idx.append(row)
        values = [value for value in list(data2.iloc[:, column]) if not math.isnan(value)]
        values = [value for value in values if not math.isinf(value)]
        for i in idx:
            data2.iloc[i, column] = np.mean(values)
    return data1, data2

def check_nan_v2(train, test):
    # 如果某个描述符有nan值，则用均值代替
    for column in range(train.shape[1]):
        if column == 0 or column == 1:
            continue
        idx = []
        for row in range(train.shape[0]):
            if math.isnan(train.iloc[row, column]) or math.isinf(train.iloc[row, column]):
                idx.append(row)
        values = [value for value in list(train.iloc[:, column]) if not math.isnan(value)]
        values = [value for value in values if not math.isinf(value)]
        for i in idx:
            train.iloc[i, column] = np.mean(values)
    for column in range(test.shape[1]):
        if column == 0 or column == 1:
            continue
        idx = []
        for row in range(test.shape[0]):
            if math.isnan(test.iloc[row, column]) or math.isinf(test.iloc[row, column]):
                idx.append(row)
        values = [value for value in list(test.iloc[:, column]) if not math.isnan(value)]
        values = [value for value in values if not math.isinf(value)]
        for i in idx:
            test.iloc[i, column] = np.mean(values)
    return train, test

def feature_selection(property, j, train_data, test_data):
    x_train, y_train = train_data.iloc[:, 2:], train_data.iloc[:, 1]
    x_test, y_test = test_data.iloc[:, 2:], test_data.iloc[:, 1]
    x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
    x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test)
    x_train, x_test = check_nan_v1(x_train, x_test)
    # 获得变量名
    feature_name = x_train.columns.values
    # 删除自身方差变化小于threshold的特征
    sel = VarianceThreshold(threshold=Variancethreshold)
    sel.fit(x_train)
    # 保存变量的选择情况
    selsuppor_t = pd.DataFrame(sel.get_support(), dtype=int)
    # 计算删除了多少变量
    before_VT_row, before_VT_col = x_train.shape
    x_train_deleted = sel.transform(x_train)
    x_test_deleted = sel.transform(x_test)
    after_VT_row, after_VT_col = x_train_deleted.shape
  #  print("VarianceThreshold删去" + str(before_VT_col - after_VT_col) + "个特征变量")
    feature_name = feature_name.T[sel.get_support()].T
    # 高相关滤波
    corrcoef = np.corrcoef(x_train_deleted.T)
    r, c = corrcoef.shape
    transform = np.array([True] * r)
    deletelist = []
    for i in range(r):
        for k in range(i + 1, r):
            if corrcoef[i][k] > Corrthreshold:
                if i not in deletelist and k not in deletelist:
                    deletelist.append(k)
                    transform[k] = False
    # 计算删除了多少变量
  #  print("相关系数高的删去" + str(len(deletelist)) + "个特征变量")
    # 保存变量的选择情况
    corrsupport = pd.DataFrame(transform, dtype=int)
    x_train_deleted_corr = x_train_deleted.T[transform].T
    x_test_deleted_corr = x_test_deleted.T[transform].T
    feature_name = feature_name.T[transform].T
    print(len(feature_name))
    pd.DataFrame(feature_name).to_csv('./data_split/{0}_feature_selection_{1}.csv'.format(property,j), index=False, header=None)
    # 保存特征选择后的数据用于模型训练
    features = ['smile', 'label']
    features.extend(feature_name)
    train = pd.read_csv(r'./data_split/{0}_train_data_{1}.csv'.format(property,j))[features]
    test = pd.read_csv(r'./data_split/{0}_test_data_{1}.csv'.format(property,j))[features]
    train, test = check_nan_v2(train, test)
    train.to_csv('./data_split/{0}_train_data_feature_selection_{1}.csv'.format(property,j), index=False)
    test.to_csv('./data_split/{0}_test_data_feature_selection_{1}.csv'.format(property,j), index=False)

def save_file(filename,property_name,properties_data):
    with open(filename, 'w', newline='') as fileout:
        filewriter = csv.writer(fileout, delimiter=',')
        for i in range(len(properties_data)):
            property_list = properties_data[i]
            if i == 0:
                filewriter.writerow(property_name)
                filewriter.writerow(property_list)
            else:
                filewriter.writerow(property_list)
        fileout.close()


df = pd.read_csv('../data/data.csv')
smis, labels = list(df.smi), list(df.label)
print('preprocess data......')
smis_new, labels_new = data_preprocess(smis, labels)    # 6665 (pos:2419,neg:4246)
'''
dict_save = {}
dict_save['smi'] = smis_new
dict_save['label'] = labels_new
pd.DataFrame(dict_save).to_csv('../data/data_preprocess.csv')
'''

########## rdkit2d ##########
print('split data......')
for i in range(1,11):
    print('start split:', i)
    property_name, properties_train, properties_test = split(smis_new, labels_new, 'rdkit2d')
    save_file('./data_split/rdkit2d_train_data_{0}.csv'.format(i),property_name,properties_train)
    save_file('./data_split/rdkit2d_test_data_{0}.csv'.format(i),property_name,properties_test)

Variancethreshold, Corrthreshold = 0.2, 0.8
for j in range(1,11):
    print('start feature selection:', j)
    train_data, test_data = pd.read_csv(r'./data_split/rdkit2d_train_data_{0}.csv'.format(j)), pd.read_csv(r'./data_split/rdkit2d_test_data_{0}.csv'.format(j))
    feature_selection('rdkit2d', j, train_data, test_data)


########## rdkit3d ##########
print('split data......')
for i in range(1,11):
    print('start split:', i)
    property_name, properties_train, properties_test = split(smis_new, labels_new, 'rdkit3d')
    save_file('./data_split/rdkit3d_train_data_{0}.csv'.format(i),property_name,properties_train)
    save_file('./data_split/rdkit3d_test_data_{0}.csv'.format(i),property_name,properties_test)

Variancethreshold, Corrthreshold = 0.2, 0.8
for j in range(1,11):
    print('start feature selection:', j)
    train_data, test_data = pd.read_csv(r'./data_split/rdkit3d_train_data_{0}.csv'.format(j)), pd.read_csv(r'./data_split/rdkit3d_test_data_{0}.csv'.format(j))
    feature_selection('rdkit3d', j, train_data, test_data)


########## ecfp4 ##########
print('split data......')
for i in range(1,11):
    print('start split:', i)
    property_name, properties_train, properties_test = split(smis_new, labels_new, 'ecfp4')
    save_file('./data_split/ecfp4_train_data_{0}.csv'.format(i),property_name,properties_train)
    save_file('./data_split/ecfp4_test_data_{0}.csv'.format(i),property_name,properties_test)

Variancethreshold, Corrthreshold = 0.2, 0.8
for j in range(1,11):
    print('start feature selection:', j)
    train_data, test_data = pd.read_csv(r'./data_split/ecfp4_train_data_{0}.csv'.format(j)), pd.read_csv(r'./data_split/ecfp4_test_data_{0}.csv'.format(j))
    feature_selection('ecfp4', j, train_data, test_data)


########## all ##########
print('split data......')
for i in range(1,11):
    print('start split:', i)
    property_name, properties_train, properties_test = split(smis_new, labels_new, 'all')
    save_file('./data_split/all_train_data_{0}.csv'.format(i),property_name,properties_train)
    save_file('./data_split/all_test_data_{0}.csv'.format(i),property_name,properties_test)

Variancethreshold, Corrthreshold = 0.2, 0.8
for j in range(1,11):
    print('start feature selection:', j)
    train_data, test_data = pd.read_csv(r'./data_split/all_train_data_{0}.csv'.format(j)), pd.read_csv(r'./data_split/all_test_data_{0}.csv'.format(j))
    feature_selection('all', j, train_data, test_data)