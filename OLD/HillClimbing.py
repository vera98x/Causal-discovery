import copy

from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.GESUtils import *
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph

import numpy as np
import time

def local_score_difference(X : np.array, graph, edge, record_local_score, name_int_mapping, parameters, score_func):
    # assumption: edge1 --> edge2
    graph_copy = copy.deepcopy(graph)
    rls = copy.deepcopy(record_local_score)

    node2 = edge.get_node2()

    PA = graph_copy.get_parents(node2)
    PAi = list(map(lambda node_PA: graph.node_map[node_PA], PA))
    i = name_int_mapping[node2.get_name()]
    delta_score = feval([score_func, X, i, PAi, parameters])
    rls[i] = delta_score

    return rls


def hill_climbing(data : np.array, graph : GeneralGraph, filename : str, column_names : List[str]):
    #initialize the score:
    start = time.time()
    X = np.mat(data)
    parameters = {}
    score_func = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)
    #score = score_g(X, graph, score_func, parameters)  # initialize the score
    maxIterations = 100
    N = X.shape[1]
    record_local_score = [0 for i in range(N)]
    name_int_mapping = {}
    score = 0

    #initialize score
    for i, node in enumerate(graph.get_nodes()):
        PA = graph.get_parents(node)
        PAi = list(map(lambda node_PA: graph.node_map[node_PA], PA))
        delta_score = feval([score_func, X, i, PAi, parameters])
        record_local_score[i] = delta_score
        score += delta_score
        name_int_mapping[node.get_name()] = i
    print("Initial score:", score)

    for i in range(maxIterations):
        print('i:', i)
        edges = graph.get_graph_edges()
        temp_score = score
        temp_graph = graph
        temp_record_local_score = record_local_score

        for edge in edges:
            graph_copy = copy.deepcopy(graph)
            graph_copy.remove_edge(edge)
            edge_record_local_score = local_score_difference(X, graph_copy, edge, record_local_score, name_int_mapping, parameters, score_func)
            score_edge = sum(edge_record_local_score)
            print("temp score", score_edge, temp_score)
            if score_edge < temp_score:
                temp_score = score_edge
                temp_graph = graph_copy
                temp_record_local_score = edge_record_local_score
        if(temp_score >= score):
            print("found optima!")
            break
        graph = temp_graph
        score = temp_score
        record_local_score = temp_record_local_score

    end = time.time()
    print()
    print("creating hill climbing is done, it took", end - start, "seconds")
    print('Score:', score)

    pdy = GraphUtils.to_pydot(graph, labels=column_names)
    pdy.write_png(filename)
    return score, graph



# print('Now start test_ges_load_linear_10_with_local_score_BIC ...')
# data_path = "Tests/data_linear_10_2.txt"
# truth_graph_path = "Tests/graph_10.txt"
# data = np.loadtxt(data_path, skiprows=1)
# truth_dag = txt2generalgraph(truth_graph_path)  # truth_dag is a GeneralGraph instance
# print(type(truth_dag))
# column_names = ['X' + str(i) for i in range(1, 21)]
# print(hill_climbing(data, truth_dag, "Results/small_test_HILL.png", column_names))