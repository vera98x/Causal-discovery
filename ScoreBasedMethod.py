from typing import Optional
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.utils.GraphUtils import GraphUtils
from df_to_tro import dfToTrainRides, TRO_matrix_to_delay_matrix
from csv_to_df import retrieveDataframe
#from createSuperGraph import get_CG_and_superGraph

import numpy as np
import time

# the goal of the ScoreBasedMethod is to first implement the supergraph using domain knowledge and then adding the backward part of the greedy search to optimize this score.

def getScore(data : np.array, result_graph : GeneralGraph) -> float:
    X = np.mat(data)
    parameters = {}
    score_func = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)
    score = score_g(X, result_graph, score_func, parameters)  # initialize the score
    return score

def initializeGES(X : numpy.matrix, score_func : str, parameters: Optional[Dict[str, Any]]) -> Tuple[LocalScoreClass, int]:
    maxP = None
    if score_func == 'local_score_marginal_multi':  # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'dlabel': {}}
            for i in range(X.shape[1]):
                parameters['dlabel'][i] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)
    elif score_func == 'local_score_marginal_general':  # negative marginal likelihood based on regression in RKHS
        parameters = {}
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)
    elif score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':  # Greedy equivalence search with BIC score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        parameters = {}
        parameters["lambda_value"] = 2
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)
    else:
        raise Exception('Unknown score function!')

    return localScoreClass, N

def backwardGES(X: ndarray, supergraph: GeneralGraph, column_names: List[str], filename : str, score_func: str = 'local_score_BIC',
        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start = time.time()

    #variables from forward search:
    if X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    X = np.mat(X)

    score_func, N = initializeGES(X, score_func, parameters)
    update1 = []
    G_step1 = [supergraph]
    G = supergraph
    record_local_score = [[] for i in range(N)]
    print("calc score:")
    start_score = time.time()
    score = score_g(X, supergraph, score_func, parameters)  # initialize the score
    end_score = time.time()
    print()
    print("creating GES initial score is done, it took", end_score - start_score, "seconds")

    print('Initial score is:', score)

    # backward greedy search
    print('backward')
    score_new = score
    update2 = []
    G_step2 = []

    while True:
        score = score_new
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if ((G.graph[j, i] == Endpoint.TAIL.value and G.graph[i, j] == Endpoint.TAIL.value)
                        or G.graph[j, i] == Endpoint.ARROW.value):  # if Xi - Xj or Xi -> Xj
                    Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
                    Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                                    np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi
                    H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
                    # for any subset of H0
                    sub = Combinatorial(H0.tolist())  # find all the subsets for H0
                    S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
                    # 1: check the condition,
                    # 2: check nothing and is valid;
                    for k in range(len(sub)):
                        if (S[k] == 1):
                            V = delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
                            if (V):
                                # find those subsets that include sub(k)
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2  # and set their S to 2
                        else:
                            V = 1

                        if (V):
                            chscore, desc, record_local_score = delete_changed_score(X, G, i, j, sub[k],
                                                                                     record_local_score, score_func,
                                                                                     parameters)
                            # calculate the changed score after Insert operator
                            # desc{count} saves the corresponding (i,j,sub{k})
                            if (chscore < min_chscore):
                                min_chscore = chscore
                                min_desc = desc

        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= 0:
                break
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            update2.append([min_desc[0], min_desc[1], min_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break
    end = time.time()
    print()
    print("creating GES backward is done, it took", end - start, "seconds")

    Record = {'update1': update1, 'update2': update2, 'G_step1': G_step1, 'G_step2': G_step2, 'G': G, 'score': score}
    pdy = GraphUtils.to_pydot(G, labels=column_names)
    pdy.write_png(filename)
    return Record


# print("extracting file")
# export_name = 'Data/6100_jan_nov_2022_2.csv' #'Data/2019-03-01_2019-05-31.csv'
# list_of_trainseries = ['6100']
# df = retrieveDataframe(export_name, True, list_of_trainseries)
# dataset_with_classes = dfToTrainRides(df)
# print("extracting file done")
#
# print("translating dataset to 2d array for algo")
# smaller_dataset = dataset_with_classes[:,:100] #np.concatenate((dataset_with_classes[:,:100], dataset_with_classes[:,300:400]), axis=1)
# # get the schedule
# sched_with_classes = smaller_dataset[0]
# res_dict = TRN_matrix_to_delay_matrix_columns_pair(smaller_dataset)
# delays_to_feed_to_algo, column_names = res_dict['delay_matrix'], res_dict['column_names']
#
# bk, cg_sched = get_CG_and_superGraph(sched_with_classes, 'Results/sched.png') #get_CG_and_background(smaller_dataset, 'Results/sched.png')
# method = 'mv_fisherz' #'fisherz'
# print("start with GES and background")
# #completeGES(delays_to_feed_to_algo, 'Results/6100_jan_nov_wo_backgr.png', 'local_score_BIC')
#
# r = backwardGES(delays_to_feed_to_algo, cg_sched.G, column_names, 'GES_test.png' 'local_score_marginal_general')

# print("starting normal GES")
# start = time.time()
# completeGES(delays_to_feed_to_algo, 'Results/6100_jan_nov_wo_backgr.png')
# end = time.time()
# print("creating SCM without background is done, it took" , end - start, "seconds")