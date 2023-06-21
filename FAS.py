import itertools
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edges import Edges
import copy
from causallearn.utils.cit import *
from typing import Dict, Tuple, List
from Utils import createIDTRNDict
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


class FAS_method():
    def __init__(self,method : str, data : np.array, filename : str, sched_with_classes: np.array, bk : BackgroundKnowledge, column_names : List[str] = None):
        self.method = method
        self.data = data
        self.filename = filename
        self.id_trn_dict = createIDTRNDict(sched_with_classes)
        self.column_names = column_names
        self.bk = bk
        self.alpha = 0.05

    def fas_(self):
        independence_test_method = CIT(self.data, method=self.method)
        nodes = []
        sepsets = {}
        adjacencies = {}
        # first create all graph nodes
        for i in range(self.data.shape[1]):
            node_name = self.column_names[i]
            node_x = GraphNode(node_name)
            node_x.add_attribute("id", i)
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
                            updated_adjacencies_x = [x for x in current_adjancencies_x if x[0] != node_j]
                            adjacencies[node_x.get_name()] = updated_adjacencies_x

                            # remove adjacency j
                            current_adjancencies_j = adjacencies.get(node_j.get_name(), [])
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
        print("start with FAS with background")
        start = time.time()
        gg_fas, sep_sets = self.fas_()
        end = time.time()
        print("FAS:", "it took", end - start, "seconds")
        gg_fas = self.orientEdges(gg_fas)
        end = time.time()
        print("creating SCM of FAS with background is done, it took", end - start, "seconds")
        if print_graph:
            pdy = GraphUtils.to_pydot(gg_fas, labels=self.column_names)
            pdy.write_png(self.filename)
        return gg_fas
