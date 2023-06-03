import itertools
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edges import Edges
import numpy as np
import warnings
import copy
from causallearn.utils.cit import *

from typing import Dict, Tuple, List
from Utils import createIDTRNDict
from FastBackgroundKnowledge import FastBackgroundKnowledge
from causallearn.utils.Fas import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.search.ConstraintBased.FCI import reorientAllWith, rule0, removeByPossibleDsep, rulesR1R2cycle, ruleR3, ruleR4B
import datetime

class FAS_method():
    def __init__(self,method : str, data : np.array, filename : str, sched_with_classes: np.array, column_names : List[str] = None, bk : BackgroundKnowledge = None):
        self.method = method
        self.data = data
        self.filename = filename
        self.id_trn_dict = createIDTRNDict(sched_with_classes)
        self.column_names = column_names
        self.bk = bk
        self.alpha = 0.05

    def fas_(self):
        independence_test_method = CIT(self.data, method=self.method) #MV_FisherZ_mod(self.data) #CIT(self.data, method=self.method)
        nodes = []
        sepsets = {}
        adjacencies = {}
        # first create all graph nodes
        for i in range(self.data.shape[1]):
            node_name = self.column_names[i]
            node_x = GraphNode(node_name)
            node_x.add_attribute("id", i)
            # trn = self.id_trn_dict[node_name]
            # node_x.add_attribute("trainnumber", trn.getTrainRideNumber())
            # node_x.add_attribute("time", trn.getPlannedTime())
            nodes.append(node_x)
        #----------------------------- Depth0 start
        print("creating nodes and dependencies...")
        node_length = len(nodes)
        for i, node_x in enumerate(nodes):
            print("Start: ", i, "/", node_length)
            for j in range(i+1, len(nodes)):
                other_node = nodes[j]
                if (self.bk.is_forbidden(nodes[i], nodes[j]) and self.bk.is_forbidden(nodes[j], nodes[i])):
                    continue
                # p_value = independence_test_method(i, j, tuple([]))
                # if(p_value > self.alpha):
                #     sepsets[(node_x.get_name(), other_node.get_name())] = []
                #     #sepsets[other_node.get_name()] = [(node_x, j)]
                else:
                    current_x = adjacencies.get(node_x.get_name(), [])
                    adjacencies[node_x.get_name()] = current_x + [(other_node, j)]
                    current_other = adjacencies.get(other_node.get_name(), [])
                    adjacencies[other_node.get_name()] = current_other + [(node_x, i)]
        # ----------------------------- Depth0 end

        # ----------------------------- Depth x start
        maxDepth = 20
        for depth in range(maxDepth):
            print("Depth: ", depth)
            enough_adjacencies_for_depth = False
            # for each depth, copy the adjacencies, such that for each depth, the dependencies are kept the same
            adjacencies_completed = copy.copy(adjacencies)
            for i, node_x in enumerate(nodes):
                print(depth, ": ", i, "/", node_length)
                adjx = adjacencies_completed.get(node_x.get_name(), [])
                if len(adjx)-1 < depth:
                    continue
                for index_j, (node_j, j) in enumerate(adjx):

                    if self.bk.is_required(node_x, node_j) or self.bk.is_required(node_j, node_x):
                        # if the node_x is required, don't try to find independences and move on to the next edge
                        continue
                    remaining_adjx = copy.copy(adjx)
                    remaining_adjx.pop(index_j)

                    cleaned_remaining_adjx = []
                    for adjx_node in remaining_adjx:
                        if not self.bk.is_required(node_x, adjx_node[0]):
                            cleaned_remaining_adjx.append(adjx_node)

                    possible_sepsets = itertools.combinations(cleaned_remaining_adjx, depth)
                    for possible_sepset in possible_sepsets:
                        enough_adjacencies_for_depth = True

                        possible_sepset = list(possible_sepset)
                        possible_sepset_indexes = [x[1] for x in possible_sepset]
                        p_value = independence_test_method(i, j, possible_sepset_indexes)
                        if (p_value > self.alpha):
                            sepsets[(node_x.get_name(), node_j.get_name())] = [possible_sepset]
                            # remove adjacency x
                            current_adjancencies_x = adjacencies.get(node_x.get_name(), [])
                            # Question: Should i remove the adjacency right away? For now I remove it directly
                            updated_adjacencies_x = [x for x in current_adjancencies_x if x[0] != node_j]
                            adjacencies[node_x.get_name()] = updated_adjacencies_x

                            # remove adjacency j
                            current_adjancencies_j = adjacencies.get(node_j.get_name(), [])
                            # Question: Should i remove the adjacency right away? For now I remove it directly
                            updated_adjacencies_j = [x for x in current_adjancencies_j if x[0] != node_x]
                            adjacencies[node_j.get_name()] = updated_adjacencies_j
            if not enough_adjacencies_for_depth:
                break
        # ----------------------------- Depth x end
        # ----------------------------- Transform to graph start
        graph = GeneralGraph(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_x = nodes[i]
                node_y = nodes[j]
                adjx = adjacencies.get(node_x.get_name(), [])
                adjx_indexes = [x[1] for x in adjx]
                if j in adjx_indexes:
                    graph.add_edge(Edges().undirected_edge(node_x, node_y))
        # ----------------------------- Transform to graph end
        return graph, sepsets




    def startupFAS(self):
        depth = 5

        if self.data.shape[0] < self.data.shape[1]:
            warnings.warn("The number of features is much larger than the sample size!")

        independence_test_method = CIT(self.data, method=self.method)

        ## ------- check parameters ------------
        if (self.bk is not None) and (type(self.bk) != BackgroundKnowledge and type(self.bk) != FastBackgroundKnowledge):
            raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
        ## ------- end check parameters ------------

        nodes = []
        for i in range(self.data.shape[1]):
            node_name = self.column_names[i]
            node = GraphNode(node_name)
            node.add_attribute("id", i)
            #trn = self.id_trn_dict[node_name]
            #node.add_attribute("trainnumber", trn.getTrainRideNumber())
            #node.add_attribute("time", trn.getPlannedTime())
            nodes.append(node)

        graph, sep_sets = fas(self.data, nodes, independence_test_method=independence_test_method, alpha=self.alpha,
                              knowledge=self.bk, depth=depth, verbose=False)
        # using fas will delete all attributes of the node, hence override the nodes with the one with the attributes
        #graph.nodes = nodes
        return graph, sep_sets

    def reorientAllWith_local(self, graph_with_edges, endpoint, number_of_nodes):
        edges = graph_with_edges.get_graph_edges()
        # empty the complete graph
        graph_with_edges.graph = np.zeros((number_of_nodes, number_of_nodes), np.dtype(int))
        graph_with_edges.dpath = np.zeros((number_of_nodes, number_of_nodes), np.dtype(int))
        for edge in edges:
            # get nodes from edge
            edge.set_endpoint1(endpoint)
            edge.set_endpoint2(endpoint)
            graph_with_edges.add_edge(edge)

        return graph_with_edges

    def orientEdges_domain_knowledge(self, graph):
        nodes = graph.get_nodes()
        num_vars = len(nodes)
        for node in nodes:
            node_name = node.get_name()
            trn_time = self.id_trn_dict[node_name].getPlannedTime_time()
            node.add_attribute('time', trn_time)
        edges = graph.get_graph_edges()
        # empty the complete graph
        graph.graph = np.zeros((num_vars, num_vars), np.dtype(int))
        # add new nodes
        for edge in edges:
            # only adjust the edges that are not defined yet
            if edge.get_endpoint1() == Endpoint.CIRCLE and edge.get_endpoint2() == Endpoint.CIRCLE:
                # get nodes from edge
                node1 = edge.get_node1()
                node2 = edge.get_node2()
                # map edges to TRN + get time
                trn1_time = node1.get_attribute('time')
                trn2_time = node2.get_attribute('time')

                reverse = False
                if (trn1_time.hour == 23 or trn2_time.hour == 23) and (trn1_time.hour == 0 or trn2_time.hour == 0):
                    reverse = True

                # order in timewise
                if trn1_time > trn2_time:
                    # add directed edge
                    if (reverse):
                        graph.add_directed_edge(node1, node2)
                    else:
                        graph.add_directed_edge(node2, node1)

                else:
                    # add directed edge
                    if (reverse):
                        graph.add_directed_edge(node2, node1)
                    else:
                        graph.add_directed_edge(node1, node2)
            else:
                graph.add_edge(edge)
        return graph

    def orientEdges_FCI(self, graph : GeneralGraph, sep_sets) -> GeneralGraph:
        nodes = graph.get_nodes()
        max_path_length = -1
        verbose = False
        reorientAllWith(graph, Endpoint.CIRCLE)

        rule0(graph, nodes, sep_sets, self.bk, True)

        removeByPossibleDsep(graph, self.method, self.alpha, sep_sets)

        self.reorientAllWith_local(graph, Endpoint.CIRCLE, len(nodes))
        rule0(graph, nodes, sep_sets, self.bk, verbose)

        change_flag = True
        first_time = True

        while change_flag:
            change_flag = False
            change_flag = rulesR1R2cycle(graph, self.bk, change_flag, verbose)
            change_flag = ruleR3(graph, sep_sets, self.bk, change_flag, verbose)

            if change_flag or (first_time and self.bk is not None and
                               len(self.bk.forbidden_rules_specs) > 0 and
                               len(self.bk.required_rules_specs) > 0):
                change_flag = ruleR4B(graph, max_path_length, self.data, self.method, self.alpha, sep_sets,
                                      change_flag,
                                      self.bk, verbose)

                first_time = False

                if verbose:
                    print("Epoch")

        graph.set_pag(True)

        return graph

    def orientEdges(self, ggFas : GeneralGraph) -> GeneralGraph:
        nodes = ggFas.get_nodes()
        num_vars = len(nodes)
        for node in nodes:
            node_name = node.get_name()
            trn_time = self.id_trn_dict[node_name].getPlannedTime_time()
            node.add_attribute('time', trn_time)
        edges = ggFas.get_graph_edges()
        # empty the complete graph
        ggFas.graph = np.zeros((num_vars, num_vars), np.dtype(int))
        # add new nodes
        for edge in edges:
            # get nodes from edge
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            # map edges to TRN + get time
            trn1_time = node1.get_attribute('time')
            trn2_time = node2.get_attribute('time')

            reverse = False
            if (trn1_time.hour == 23 or trn2_time.hour == 23) and (trn1_time.hour == 0 or trn2_time.hour == 0):
                reverse = True

            #order in timewise
            if trn1_time > trn2_time:
                #add directed edge
                if(reverse):
                    ggFas.add_directed_edge(node1, node2)
                else:
                    ggFas.add_directed_edge(node2, node1)

            else:
                # add directed edge
                if (reverse):
                    ggFas.add_directed_edge(node2, node1)
                else:
                    ggFas.add_directed_edge(node1, node2)
        return ggFas

    def fas_with_background(self, print_graph) -> GeneralGraph:
        with_or_without = "with" if self.bk != None else "without"
        print("start with FAS " + with_or_without + " background")
        start = time.time()
        gg_fas, sep_sets = self.fas_()
        end = time.time()
        print("FAS:", "it took", end - start, "seconds")
        gg_fas = self.orientEdges(gg_fas)
        #gg_fas = self.orientEdges_domain_knowledge(gg_fas)
        end = time.time()
        print("creating SCM of FAS " + with_or_without + " background is done, it took", end - start, "seconds")
        if print_graph:
            pdy = GraphUtils.to_pydot(gg_fas, labels=self.column_names)
            pdy.write_png(self.filename)
        return gg_fas

# class MV_FisherZ_mod(CIT_Base):
#     def __init__(self, data, **kwargs):
#         super().__init__(data, **kwargs)
#         self.check_cache_method_consistent('mv_fisherz', NO_SPECIFIED_PARAMETERS_MSG)
#         self.assert_input_data_is_valid(allow_nan=True)
#
#     def corr_coef_approach(self, x_matrix):
#         # source: https://stackoverflow.com/questions/21444135/how-to-improve-very-inefficient-numpy-code-for-calculating-correlation
#         final_array = np.array([])
#         demeaned = x_matrix - x_matrix.mean(axis=1)[:, None]
#         for index in range(len(x_matrix)):
#             # Dot product of each row with index
#             res = np.ma.dot(demeaned, demeaned[index])
#             # Norm of each row
#             row_norms = np.ma.sqrt((demeaned ** 2).sum(axis=1))
#             # Normalize
#             res = res / row_norms / row_norms[index]
#             if index == 0:
#                 final_array = [res]
#             else:
#                 final_array = np.concatenate((final_array, [res]))
#         return final_array
#
#     def __call__(self, X, Y, condition_set=None):
#         '''
#         Perform an independence test using Fisher-Z's test for data with missing values.
#
#         Parameters
#         ----------
#         X, Y and condition_set : column indices of data
#
#         Returns
#         -------
#         p : the p-value of the test
#         '''
#         if condition_set is None:
#             condition_set = []
#         Xs = [X]
#         Ys = [Y]
#         cache_key_small = str(X) + ";" + str(Y)
#         _strlst = lambda lst: '.'.join(map(str, lst))
#         cache_key = cache_key_small if len(condition_set) == 0 else cache_key_small + "|" + _strlst(condition_set)
#         #Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
#         if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
#         var = Xs + Ys + condition_set
#         data_with_correct_columns = self.data[:, var]
#         test_wise_deleted_data_var = data_with_correct_columns[~np.isnan(data_with_correct_columns).any(axis=1)]
#         #test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(self.data[:, var])
#         assert len(test_wise_deleted_data_var) != 0, \
#             "A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data."
#         #test_wise_deleted_data_var = self.data[test_wise_deletion_XYcond_rows_index][:, var]
#         #sub_corr_matrix = self.corr_coef_approach(test_wise_deleted_data_var.T)
#         sub_corr_matrix = np.corrcoef(test_wise_deleted_data_var.T)
#         try:
#             inv = np.linalg.inv(sub_corr_matrix)
#         except np.linalg.LinAlgError:
#             raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
#         r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
#         Z = 0.5 * log((1 + r) / (1 - r))
#         X = sqrt(len(test_wise_deleted_data_var) - len(condition_set) - 3) * abs(Z)
#         p = 2 * (1 - norm.cdf(abs(X)))
#         self.pvalue_cache[cache_key] = p
#         return p
