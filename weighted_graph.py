# coding=utf-8

import pandas as pd
import numpy as np

import networkx as nx
from collections import defaultdict
from itertools import permutations
from itertools import combinations
import gensim
import gensim.downloader as api
import community as comg
from sklearn.metrics import mean_absolute_error

threshold = 0  # Threshold for building edges


dataname = 'datasets/anew.csv'  # Training set file
vectormodel = 'word2vec.model'  # word2vec model file

data = pd.read_csv(dataname,  # Read in the training set
                   sep=',',
                   encoding='utf-8')
model = api.load("word2vec-google-news-300")  # Load in model


# Data segmentation
def data_split(data, ratio=0.8):
    '''
    Dataset split：Split the data set into a training set and a test set according to ratio
    :param data: All data
    :param ratio: split ratio
    :return: training set and test set
    '''
    np.random.seed(1)
    size = int(len(data) * ratio)  # Training set length
    shuffle = list(range(len(data)))  # Training set index
    np.random.shuffle(shuffle)  # Randomly shuffle index

    train = data.iloc[shuffle[:size]]
    test = data.iloc[shuffle[size:]]
    return train, test


train, test = data_split(data)


# get the VA pair of dataset
def data_dict(data):
    word_VA = {}
    data = data[['text', 'valence', 'arousal']].values

    for word, V, A in data:
        word_VA[word] = [V, A]
    return word_VA


word_VA = data_dict(train)


class Louvain(object):
    def __init__(self):
        self.MIN_VALUE = 0.0000001
        self.node_weights = {}

    @classmethod
    def convertIGraphToNxGraph(cls, igraph):
        node_names = igraph.vs["name"]
        edge_list = igraph.get_edgelist()
        weight_list = igraph.es["weight"]
        node_dict = defaultdict(str)

        for idx, node in enumerate(igraph.vs):
            node_dict[node.index] = node_names[idx]

        convert_list = []
        for idx in range(len(edge_list)):
            edge = edge_list[idx]
            new_edge = (node_dict[edge[0]], node_dict[edge[1]], weight_list[idx])
            convert_list.append(new_edge)

        convert_graph = nx.Graph()
        convert_graph.add_weighted_edges_from(convert_list)
        return convert_graph

    def updateNodeWeights(self, edge_weights):
        node_weights = defaultdict(float)
        for node in edge_weights.keys():
            node_weights[node] = sum([weight for weight in edge_weights[node].values()])
        return node_weights

    def getBestPartition(self, graph, param=1.):
        node2com, edge_weights = self._setNode2Com(graph)

        node2com = self._runFirstPhase(node2com, edge_weights, param)
        best_modularity = self.computeModularity(node2com, edge_weights, param)

        partition = node2com.copy()
        new_node2com, new_edge_weights = self._runSecondPhase(node2com, edge_weights)

        while True:
            new_node2com = self._runFirstPhase(new_node2com, new_edge_weights, param)
            modularity = self.computeModularity(new_node2com, new_edge_weights, param)
            if abs(best_modularity - modularity) < self.MIN_VALUE:
                break
            best_modularity = modularity
            partition = self._updatePartition(new_node2com, partition)
            _new_node2com, _new_edge_weights = self._runSecondPhase(new_node2com, new_edge_weights)
            new_node2com = _new_node2com
            new_edge_weights = _new_edge_weights
        return partition

    def computeModularity(self, node2com, edge_weights, param):
        q = 0
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2

        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)

        for com_id, nodes in com2node.items():
            node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes]
            cluster_weight = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations])
            tot = self.getDegreeOfCluster(nodes, node2com, edge_weights)
            q += (cluster_weight / (2 * all_edge_weights)) - param * ((tot / (2 * all_edge_weights)) ** 2)
        return q

    def getDegreeOfCluster(self, nodes, node2com, edge_weights):
        weight = sum([sum(list(edge_weights[n].values())) for n in nodes])
        return weight

    def _updatePartition(self, new_node2com, partition):
        reverse_partition = defaultdict(list)
        for node, com_id in partition.items():
            reverse_partition[com_id].append(node)

        for old_com_id, new_com_id in new_node2com.items():
            for old_com in reverse_partition[old_com_id]:
                partition[old_com] = new_com_id
        return partition

    def _runFirstPhase(self, node2com, edge_weights, param):
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        self.node_weights = self.updateNodeWeights(edge_weights)
        status = True
        while status:
            statuses = []
            for node in node2com.keys():
                statuses = []
                com_id = node2com[node]
                neigh_nodes = [edge[0] for edge in self.getNeighborNodes(node, edge_weights)]

                max_delta = 0.
                max_com_id = com_id
                communities = {}
                for neigh_node in neigh_nodes:
                    node2com_copy = node2com.copy()
                    if node2com_copy[neigh_node] in communities:
                        continue
                    communities[node2com_copy[neigh_node]] = 1
                    node2com_copy[node] = node2com_copy[neigh_node]

                    delta_q = 2 * self.getNodeWeightInCluster(node, node2com_copy, edge_weights) - (self.getTotWeight(
                        node, node2com_copy, edge_weights) * self.node_weights[node] / all_edge_weights) * param
                    if delta_q > max_delta:
                        max_delta = delta_q
                        max_com_id = node2com_copy[neigh_node]

                node2com[node] = max_com_id
                statuses.append(com_id != max_com_id)

            if sum(statuses) == 0:
                break

        return node2com

    def _runSecondPhase(self, node2com, edge_weights):
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda: defaultdict(float))

        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
        for edge in node_pairs:
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][
                edge[1]]
        return new_node2com, new_edge_weights

    def getTotWeight(self, node, node2com, edge_weights):
        nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    def getNeighborNodes(self, node, edge_weights):
        if node not in edge_weights:
            return 0
        return edge_weights[node].items()

    def getNodeWeightInCluster(self, node, node2com, edge_weights):
        neigh_nodes = self.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]
        return weights

    def _setNode2Com(self, graph):
        node2com = {}
        edge_weights = defaultdict(lambda: defaultdict(float))
        for idx, node in enumerate(graph.nodes()):
            node2com[node] = idx
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]["weight"]
        return node2com, edge_weights


def makeSampleGraph():
    g = nx.Graph()

    data_word = data.text.values
    size = len(data_word)
    for i in range(size - 1):
        if(i%100 ==0):
            print('100 completed')
        for j in range(i + 1, size):
            if (data_word[i] in model and data_word[j] in model):
                sim = model.similarity(data_word[i], data_word[j])
                if (sim > threshold):
                    g.add_edge(data_word[i], data_word[j], weight=sim)
    return g


# Calculate the unseen word. After moving to a specific community cluster, only among the words connected to the unseen word, summarize their contribution VA value to the unseen word by weight.
def calculate_VA(word, clusters, word_VA):
    weighted_VA = 0
    totle_sims = 0
    ok = False

    for word1 in clusters:
        if (word1 not in word_VA.keys()):
            continue
        sim = model.similarity(word, word1)
        if sim >= threshold:
            ok = True
            tmp = np.array(word_VA[word1])
            weighted_VA += tmp * sim
            totle_sims += sim
        ok = True
        tmp = np.array(word_VA[word1])
        weighted_VA += tmp * sim
        totle_sims += sim

    if (ok == False):  # If there is no seed VA word in cluster, return [5, 5]
        return [5, 5],False
    return weighted_VA / totle_sims,True


# Evaluation
def evaluation(test, result):
    n = len(test)  # Test set size
    V_mae, V_r = 0, 0  # Valence Mean Absolute Error, Pearson Coefficient
    A_mae, A_r = 0, 0  # Arousal Mean Absolute Error, Pearson Coefficient
    V_A_mean, A_A_mean = np.mean(test, axis=0)  # True VA mean
    V_P_mean, A_P_mean = np.mean(result, axis=0)  # Predicted VA mean
    V_std, A_std = np.std(test, axis=0)  # VA variance
    V_P_std, A_P_std = np.std(result, axis=0)  # VA mean

    print ('V_A_mean:', V_A_mean, '-------', 'V_P_mean:', V_P_mean)
    print ('A_A_mean:', A_A_mean, '-------', 'A_P_mean:', A_P_mean)
    print
    print ('V_std:', V_std, '-------', 'V_P_std:', V_P_std)
    print ('A_std:', A_std, '-------', 'A_P_std:', A_P_std)
    print
    print

    # Calculate test indicators according to official methods
    for i in range(n):
        V_mae += np.abs(test[i, 0] - result[i, 0])
        A_mae += np.abs(test[i, 1] - result[i, 1])
        V_r += ((test[i, 0] - V_A_mean) / V_std) * ((result[i, 0] - V_P_mean) / V_P_std)
        A_r += ((test[i, 1] - A_A_mean) / A_std) * ((result[i, 1] - A_P_mean) / A_P_std)
    print ('Valence mean absolute error:', V_mae / n)
    print ('Arousal mean absolute error:', A_mae / n)
    print ('Valence pearson correlation coefficient:', V_r / (n - 1))
    print ('Arousal pearson correlation coefficient:', A_r / (n - 1))


def run():
    sample_graph = makeSampleGraph()
    louvain = Louvain()
    partition = louvain.getBestPartition(sample_graph)
    p = defaultdict(list)

    for node, com_id in partition.items():
        p[com_id].append(node)

    cnt = len(data)
    nodes2word = [[] for i in range(cnt)]
    num = 0
    for com, nodes in p.items():
        for n in nodes:
            nodes2word[com].append(n)
            num += 1

    print('num',num)
    word2nodes = {}
    for i in range(cnt):
        for word in nodes2word[i]:
            word2nodes[word] = i

    V_array = np.asarray(train['valence'].values.tolist() + [5] * test.shape[0])
    A_array = np.asarray(train['arousal'].values.tolist() + [5] * test.shape[0])
    all_words = train['text'].values.tolist() + test['text'].values.tolist()

    num_all_words = len(all_words)
    num_train_words = len(train)
    num_test_words = len(test)

    print(len(all_words))
    S = np.zeros((num_all_words, num_all_words))
    for word_idx, word in enumerate(all_words):
        words_sim = [
            model.similarity(word, item) if word2nodes.get(item, -1) == -1 or word2nodes.get(
                word, -1) == -1 or word2nodes[word] == word2nodes[item] else 0 for item in all_words]
            
        S[word_idx] = words_sim

    alpha = 0.1

    D = np.asarray([0] * num_train_words + [alpha] * num_test_words)
    I = np.ones(num_all_words)
    Vt = V_array
    At = A_array
    num_epoch = 50

    for epoch in range(1, num_epoch + 1):
        Vt = ((I - D) * Vt) + (D * (np.dot(S, Vt) / np.dot(S, I)))
        Vt = np.nan_to_num(Vt)

        At = ((I - D) * At) + (D * (np.dot(S, Vt) / np.dot(S, I)))
        At = np.nan_to_num(At)

        print('Iteration %d' % epoch)
        print(Vt[-num_test_words:])
        print(At[-num_test_words:])

    test_V_pred = Vt[-num_test_words:]
    test_A_pred = At[-num_test_words:]

    print(test_V_pred)
    print(test['valence'] - test_V_pred)

    V_mae = mean_absolute_error(test['valence'].values, test_V_pred)
    A_mae = mean_absolute_error(test['arousal'].values, test_A_pred)
    print(V_mae)
    print(A_mae)



    test_words = test[['text', 'valence', 'arousal']].values
    test1 = test[['valence', 'arousal']].values
    size_test = len(test1)
    result = []
    none = 0
    for i in range(size_test):
        word = test_words[i, 1]
        print(word2nodes.keys())
        if (word not in word2nodes.keys()):
            result.append([5, 5])
            print (word, [5, 5])
            none += 1
            continue
        print(nodes2word[word2nodes[word]])
        ans,falg = calculate_VA(word, nodes2word[word2nodes[word]], word_VA)
        result.append(ans)
        if(falg==False):
            none +=1
        print (word, ans)

    print ('Not found in word2vec, or there is no reference value within the cluster, or the highest similarity cannot reach the threshold', none)
    result = np.array(result).round(1)  # 1 decimal place
    evaluation(test1, result)
    total = len(test1)
    v_inverse = 0
    a_inverse = 0
    mae_v = 0
    rmse_v = 0
    mae_a = 0
    rmse_a = 0
    for i in range(len(test)):
        if test_words[i, 1] > 5:
            if result[i, 0] < 5:
                v_inverse += 1
        else:
            if result[i, 0] > 5:
                v_inverse += 1
        if test_words[i, 2] > 5:
            if result[i, 0] < 5:
                v_inverse += 1
        else:
            if result[i, 0] > 5:
                v_inverse += 1
        mae_v += abs(test_words[i, 1] - result[i, 0])
        mae_a += abs(test_words[i, 2] - result[i, 1])
        rmse_v += (test_words[i, 1] - result[i, 0]) * (test_words[i, 1] - result[i, 0])
        rmse_a += (test_words[i, 2] - result[i, 1]) * (test_words[i, 2] - result[i, 1])
    mae_v = np.sqrt(mae_v / total)
    mae_a = np.sqrt(mae_a / total)
    rmse_v = np.sqrt(rmse_v / total)
    rmse_a = np.sqrt(rmse_a / total)
    print ('the mae of V :', mae_v, 'the mae of A :', mae_a)
    print ('the rmse of V :', rmse_v, 'the rmse of A :', rmse_a)
    print ('the Inverse Polarity of V is : ', v_inverse * 0.1 / total)
    print ('the Inverse Polarity of V is : ', a_inverse * 0.1 / total)
    df = pd.DataFrame({'No.': test_words[:, 0]})
    df['WORD'] = test_words[:, 0]
    df['TRUE_V'] = test_words[:, 1]
    df['PREDICT_V'] = result[:, 0]
    df['V_ERROR'] = test_words[:, 2] - result[:, 0]
    df['TRUE_A'] = test_words[:, 2]
    df['PREDICT_A'] = result[:, 1]
    df['A_ERROR'] = test_words[:, 2] - result[:, 1]

    df.to_csv('result_processing.csv',
              index=None,
              sep=',',
              encoding='utf-8')
    return


run()
