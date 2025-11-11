"""数据处理的工具"""
import csv
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle


def getOpcodeLabel(name=None):
    if name == None:
        vul_data1 = pd.read_csv("../dataset/simpleOpcode/reentrancy.csv")
        vul_data2 = pd.read_csv("../dataset/simpleOpcode/timestamp.csv")
        vul_data3 = pd.read_csv("../dataset/simpleOpcode/delegatecall.csv")
        vul_data4 = pd.read_csv("../dataset/simpleOpcode/SBunchecked_low_level_calls.csv")
        vul_data = pd.concat([vul_data1, vul_data2, vul_data3, vul_data4], axis=0)
        opcode = vul_data['opcode'].values
        label = vul_data['label'].values
    else:
        vul_data = pd.read_csv('../dataset/simpleOpcode/'+name+'.csv')
        opcode = vul_data['opcode'].values
        label = vul_data['label'].values
    return opcode, label

def getdataset(name):
    if name == "mix_vulnerabilities":
        vul_data1 = pd.read_csv("./dataset/embedding/smart_contract/reentrancy.csv")
        vul_data2 = pd.read_csv("./dataset/embedding/smart_contract/timestamp.csv")
        vul_data3 = pd.read_csv("./dataset/embedding/smart_contract/delegatecall.csv")
        vul_data4 = pd.read_csv("./dataset/embedding/smart_contract/SBunchecked_low_level_calls.csv")
        vul_data = pd.concat([vul_data1, vul_data2, vul_data3, vul_data4], axis=0).values
    else:
        vul_data = pd.read_csv("E:/Py_projects/SCG/dataset/embedding/smart_contract/" + name + ".csv").values[:10000]
    print("train_data.shape", vul_data.shape)
    return vul_data


def getdataset2(name, generated_num):
    normal_data = pd.read_csv("../dataset/embedding/smart_contract/normal.csv")
    normal_label = np.zeros(normal_data.shape[0])
    vul_data = pd.read_csv("../dataset/embedding/smart_contract/" + name + ".csv")
    vul_label = np.ones(vul_data.shape[0])
    generate_data = pd.read_csv("../dataset/embedding/generated_contract/generated_" + name + "_with_sem_gan.csv",
                                index_col=0).iloc[:generated_num]
    generate_label = np.ones(generate_data.shape[0])

    X_train = pd.concat([normal_data[:-vul_data.shape[0]], vul_data[:vul_data.shape[0] // 2], generate_data], axis=0)
    y_train = pd.concat([pd.Series(normal_label[:-vul_data.shape[0]]), pd.Series(vul_label[:vul_data.shape[0] // 2]),
                         pd.Series(generate_label)], axis=0).values
    X_test = pd.concat([normal_data[-vul_data.shape[0]:], vul_data[vul_data.shape[0] // 2:]], axis=0)
    y_test = pd.concat([pd.Series(normal_label[-vul_data.shape[0]:]), pd.Series(vul_label[vul_data.shape[0] // 2:])],
                       axis=0).values

    # data_x1 = pd.concat([normal_data, vul_data], axis=0)
    # data_y1 = pd.concat([pd.Series(normal_label), pd.Series(vul_label)], axis=0)
    # X_train1, X_test, y_train1, y_test = train_test_split(data_x1, data_y1, test_size=0.3, random_state=1)
    #
    # X_train = pd.concat([X_train1, generate_data], axis=0)
    # y_train = pd.concat([y_train1, pd.Series(generate_label)], axis=0).values

    y_train = y_train.ravel()  # 为了让其shape为 [n, 1], 而不是[n]
    y_test = y_test.ravel()

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def getdataset3(name, generated_num):
    # 读取训练数据
    normal_data = pd.read_csv("../dataset/embedding/smart_contract/normal.csv")
    normal_label = np.zeros(1500)
    vul_data1 = pd.read_csv("../dataset/embedding/smart_contract/reentrancy.csv")
    vul_data2 = pd.read_csv("../dataset/embedding/smart_contract/timestamp.csv")
    vul_data3 = pd.read_csv("../dataset/embedding/smart_contract/delegatecall.csv")
    vul_data4 = pd.read_csv("../dataset/embedding/smart_contract/SBunchecked_low_level_calls.csv")
    unknown_data = pd.read_csv("../dataset/embedding/generated_contract/generated_" + name + ".csv", index_col=0).iloc[:generated_num]
    test_normal_label = np.zeros(150)
    vul_label = np.ones(vul_data1.shape[0] + vul_data2.shape[0] + vul_data3.shape[0] +
                        vul_data4.shape[0] + unknown_data.shape[0])
    # vul_label = np.concatenate([test_normal_label, vul_label])


    X_train = pd.concat([normal_data[:1500], vul_data1, vul_data2, vul_data3, vul_data4, unknown_data], axis=0)
    y_train = pd.concat([pd.Series(normal_label), pd.Series(vul_label)], axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1)

    # 读取测试数据
    unknown_data1 = pd.read_csv("../dataset/embedding/smart_contract/SBaccess_control.csv")
    unknown_data2 = pd.read_csv("../dataset/embedding/smart_contract/SBarithmetic.csv")
    unknown_data3 = pd.read_csv("../dataset/embedding/smart_contract/SBdenial_of_service.csv")
    unknown_data4 = pd.read_csv("../dataset/embedding/smart_contract/SBshort_address.csv")
    unknown_data5 = pd.read_csv("../dataset/embedding/smart_contract/SBunchecked_low_level_calls.csv")
    # X_test = pd.concat([normal_data.iloc[500:520], unknown_data1, unknown_data2, unknown_data3,
    #                     unknown_data4, unknown_data5], axis=0)
    # vul_label = np.ones(unknown_data1.shape[0] + unknown_data2.shape[0] + unknown_data3.shape[0] +
    #                     unknown_data4.shape[0] + unknown_data5.shape[0])
    # y_test = pd.concat([pd.Series(normal_label[:20]), pd.Series(vul_label)], axis=0)
    X_test = pd.concat([normal_data[-1500:]
                           # , unknown_data1
                           # , unknown_data2
                           # , unknown_data3
                        #, unknown_data4
                        , unknown_data5
                        ],
                       axis=0)
    test_normal_label = np.zeros(1500)
    unknown_label = np.ones(
        # unknown_data1.shape[0] +
        # unknown_data2.shape[0] +
        #                     unknown_data3.shape[0]
                        # + unknown_data4.shape[0] +
                           unknown_data5.shape[0]
                            )
    y_test = np.concatenate([test_normal_label, unknown_label])

    y_train = y_train.values.ravel()  # 为了让其shape为 [3019, 1] ，而不是[3019]
    y_test = y_test.ravel()

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def getsequence(X_train, max_len):
    # 导入训练好的Tokenizer
    with open('E:/Py_projects/SCG/data_processing/tok.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    from keras_preprocessing import sequence
    # 对每个词编码之后，每个句子中的每个词就可以用对应的编码表示，即每个句子可以转变成一个向量了
    train_seq = tok.texts_to_sequences(X_train)
    # test_seq = tok.texts_to_sequences(X_test)
    # 将每个序列调整为相同的长度
    train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
    # test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)
    print("train_seq_mat", train_seq_mat.shape)
    # print("test_seq_mat", test_seq_mat.shape)
    return train_seq_mat  # , test_seq_mat


def word_information(tok):
    """查看编码后word的信息"""
    # 使用word_index属性可以看到每次词对应的编码
    # 使用word_counts属性可以看到每个词对应的频数
    for ii, iterm in enumerate(tok.word_index.items()):
        if ii < 10:
            print(iterm)
        else:
            break
    print("===================")
    for ii, iterm in enumerate(tok.word_counts.items()):
        if ii < 10:
            print(iterm)
        else:
            break

def opcode2embedding(name):
    """利用tokenizer得到opcode sequence的embedding"""
    # 保存位置
    savepath = '../dataset/embedding/smart_contract/' + name + ".csv"
    # 参数设置 parameters
    max_len = 500  # 句子的最大长度

    # 读取数据
    filepath = "../dataset/simpleOpcode/" + name + ".csv"
    data = pd.read_csv(filepath)
    opcode = data["opcode"].values

    # 导入训练好的Tokenizer
    import pickle
    with open('E:\Py_projects\SCG/data_processing/tok.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    # 查看重新导入的tok结果：
    word_information(tok)

    from keras_preprocessing import sequence
    # 对每个词编码之后，每个句子中的每个词就可以用对应的编码表示，即每个句子可以转变成一个向量了：
    embedding = tok.texts_to_sequences(opcode)
    # 将每个序列调整为相同的长度
    embedding_mat = sequence.pad_sequences(embedding, maxlen=max_len)
    print(embedding_mat.shape)

    df = pd.DataFrame(embedding_mat)
    df.to_csv(savepath, index=False)
    print("合约信息写入csv!")

    # list为CSV的表头
    # list = ['SC', 'label', "embedding"]
    # for i in range(max_len):
    #     list.append("i")

    # 将每个合约的embedding作为一个字典，加入list1中
    # list1 = []
    # for i in range(SCname.shape[0]):
    #     tmp = {}
    #     tmp['SC'] = SCname[i]
    #     tmp['label'] = label[i]
    #     tmp['embedding'] = " ".join(map(str, opcode_seq_mat[i]))
    #     list1.append(tmp)
    # headers = list
    # with open(savepath, 'w', newline='') as s:
    #     # s_csv = csv.writerow(headers)
    #     s_csv = csv.DictWriter(s, headers)
    #     s_csv.writeheader()
    #     s_csv.writerows(list1)

def embedding2opcode(embedding):
    # 加载训练好的tok
    with open('E:\\Py_projects\\SCG\\data_processing\\tok.pickle', 'rb') as handle:
    # with open('../../data_processing/tok.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    # print(embedding)
    text_sequences = tok.sequences_to_texts(embedding)
    # print(text_sequences)
    return text_sequences

def get_Bigram(opcode_sequence):
    # 将bytecode_sequence转化为opcode_sequence
    opcode_sequence = embedding2opcode(opcode_sequence)

    # 生成Bigram features
    from nltk import ngrams
    lines = opcode_sequence[0].split()
    ngram = ngrams(lines, 2)

    # 用字典保存结果, 生成语义特征序列
    with open("gram.txt") as txt:
        str1 = txt.read()
        dic = eval(str1.strip('"'))  # 把gram.txt转换为字典
    sumg = 0
    for i in ngram:
        if i in dic.keys():
            dic[i] += 1
        sumg += 1
    tmp = []
    for key in dic:
        tmp.append(dic[key]/sumg)
    # print(tmp)
    return tmp


def visionable(test_pre, ytest):
    """混淆矩阵可视化"""
    # 评价预测效果，计算混淆矩阵
    from sklearn import metrics
    confm = metrics.confusion_matrix(np.argmax(test_pre, axis=1), np.argmax(ytest, axis=1))
    # 混淆矩阵可视化
    Labname = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
    plt.figure(figsize=(8, 8))
    sns.heatmap(confm.T, square=True, annot=True,
                fmt='d', cbar=False, linewidths=.8,
                cmap="YlGnBu")
    plt.xlabel('True label', size=14)
    plt.ylabel('Predicted label', size=14)
    plt.xticks(np.arange(10) + 0.5, Labname)
    plt.yticks(np.arange(10) + 0.3, Labname)
    plt.show()

def label_num(ytrain):
    """图示训练集都有哪些标签"""
    plt.figure()
    sns.countplot(x=ytrain)
    plt.xlabel('Label')
    plt.xticks()
    plt.show()

