import itertools
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edges import Edges
import copy
from causallearn.utils.cit import *
from typing import Dict, Tuple, List
from Utils import createIDTRODict
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


class PCAndBackground():
    def __init__(self, method : str, data : np.array, tro_schedule_list: np.array, bk : BackgroundKnowledge):
        self.method = method
        self.data = data
        self.id_tro_dict = createIDTRODict(tro_schedule_list)
        self.column_names = column_names = np.array(list(map(lambda x: x.getSmallerID(), tro_schedule_list)))
        self.bk = bk
        self.alpha = 0.05

    def fas_(self):
        '''The creation of the FAS function is inspired by the code of the causal learn library: https://github.com/py-why/causal-learn/blob/0.1.3.0/causallearn/utils/Fas.py'''
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
        # only add the nodes that are not forbiddden in the adjacency list
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
        # prune the edges from depth 0 by performing independence tests (but remain the required edges)
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
                    # remove the other node as possible set to condition on
                    remaining_adjx.pop(index_j)

                    cleaned_remaining_adjx = []
                    for adjx_node in remaining_adjx:
                        # do not add required nodes in the adjacency list This was also done in the causal lib:
                        # https://github.com/py-why/causal-learn/blob/0.1.3.0/causallearn/utils/Fas.py#L18
                        if not self.bk.is_required(node_x, adjx_node[0]):
                            cleaned_remaining_adjx.append(adjx_node)
                    # create all sets of length dept as possible sepset
                    possible_sepsets = itertools.combinations(cleaned_remaining_adjx, depth)
                    for possible_sepset in possible_sepsets:
                        enough_adjacencies_for_depth = True

                        possible_sepset = list(possible_sepset)
                        possible_sepset_indexes = [x[1] for x in possible_sepset]
                        p_value = independence_test_method(i, j, possible_sepset_indexes)
                        if (p_value > self.alpha):
                            # add the possible sepset as sepset for the nodes
                            sepsets[(node_x.get_name(), node_j.get_name())] = [possible_sepset]
                            # remove adjacency x
                            current_adjancencies_x = adjacencies.get(node_x.get_name(), [])
                            updated_adjacencies_x = [x for x in current_adjancencies_x if x[0] != node_j]
                            adjacencies[node_x.get_name()] = updated_adjacencies_x

                            # remove adjacency j
                            current_adjancencies_j = adjacencies.get(node_j.get_name(), [])
                            updated_adjacencies_j = [x for x in current_adjancencies_j if x[0] != node_x]
                            adjacencies[node_j.get_name()] = updated_adjacencies_j
            # if the nodes are already separated by one sepset, do not search further.
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
        # for all nodes, add the tiem attribute, such that it does not need to be retrieved every time
        for node in nodes:
            node_name = node.get_name()
            tro_time = self.id_tro_dict[node_name].getPlannedTime_time()
            node.add_attribute('time', tro_time)
        edges = ggFas.get_graph_edges()
        # empty the complete graph
        ggFas.graph = np.zeros((num_vars, num_vars), np.dtype(int))
        # add new nodes
        for edge in edges:
            # get nodes from edge
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            # map edges to TRO + get time
            tro1_time = node1.get_attribute('time')
            tro2_time = node2.get_attribute('time')

            reverse = False
            # on the same date, 00h is later than 23h, so then we need to reverse the edges.
            if (tro1_time.hour == 23 or tro2_time.hour == 23) and (tro1_time.hour == 0 or tro2_time.hour == 0):
                reverse = True

            #order in timewise
            if tro1_time > tro2_time:
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

    def apply_pc_with_background(self, graph_save_png_bool, filename = None) -> GeneralGraph:
        '''The PC-algorithm is applied
        first, it performs a skeleton search using the self.fas_() function
        then it orients the edges.
        Per step, the timing is captured and lastly, the graph can be saved as an image if graph_save_png_bool is true'''
        if(graph_save_png_bool == True and filename == None):
            raise ValueError("Filename is not provided, but print_graph_bool is True")
        print("start with FAS with background")
        start = time.time()
        gg_fas, sep_sets = self.fas_()
        end = time.time()
        print("FAS:", "it took", end - start, "seconds")
        gg_fas = self.orientEdges(gg_fas)
        end = time.time()
        print("creating SCM of FAS with background is done, it took", end - start, "seconds")
        if graph_save_png_bool:
            pdy = GraphUtils.to_pydot(gg_fas, labels=self.column_names)
            pdy.write_png(filename)
        return gg_fas
