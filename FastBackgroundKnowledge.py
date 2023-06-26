from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.Node import Node
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from typing import List
class FastBackgroundKnowledge(BackgroundKnowledge):
    '''The FastBackgroundKnowledge class is a subclass of the BackgroundKnowledge class of the causal-learn library:
    https://github.com/py-why/causal-learn/blob/0.1.3.0/causallearn/utils/PCUtils/BackgroundKnowledge.py
    Some functions are overwritten, but the code is still inspired/copied from the library.'''
    def __init__(self):
        super().__init__()
        self.dict_forbidden = {}
        self.dict_required = {}

    def addForbiddenDependency_dict(self, node1_name: str, node2_names: List[str]) -> None:
        '''Adds the name of node 2 of the list of forbidden dependencies'''
        dep_list = self.dict_forbidden.get(node1_name, [])
        self.dict_forbidden[node1_name] = dep_list + node2_names

    def addDependency_dict(self, node1_name: str, node2_name: str) -> None:
        '''Adds the name of node 2 of the list of required dependencies,
        It also removes the forbidden dependency, else this would be conflicting'''
        dep_list = self.dict_required.get(node1_name, [])
        self.removeForbiddenDependency_dict(node1_name, node2_name)
        self.dict_required[node1_name] = dep_list + [node2_name]
    def removeForbiddenDependency_dict(self, node1_name: str, node2_name: str) -> None:
        '''Removes the name of node 2 of the list of forbidden dependencies'''
        dep_list = self.dict_forbidden.get(node1_name, [])
        newdep = list(filter(lambda a: a != node2_name, dep_list))
        self.dict_forbidden[node1_name] = newdep

    def backgroundToGraph(self, column_names: List[str]) -> CausalGraph:
        '''A causal graph is made from the background knowledge
        Parameter: column names in a specific order
        Return type: CausalGraph'''
        # create all nodes
        nodes = [GraphNode(i) for i in column_names]
        node_dict = {}
        for index in range(len(nodes)):
            node_dict[column_names[index]] = nodes[index]

        # form to CausalGraph
        cg = CausalGraph(len(column_names), nodes)
        # It is not possible to add edges to the CG, so create GG and add the edges that are found in the required_rules_specs
        gg = GeneralGraph(nodes)
        for key, values in self.dict_required.items():
            for value in values:
                gg.add_directed_edge(node_dict[key], node_dict[value])
        cg.G = gg
        return cg

    def is_forbidden(self, node1: Node, node2: Node) -> bool:
        """
        check whether the edge node1 --> node2 is forbidden

        Parameters
        ----------
        node1: the from node in edge which is checked
        node2: the to node in edge which is checked

        Returns
        -------
        if the  edge node1 --> node2 is forbidden, then return True, otherwise False.
        """
        values = self.dict_forbidden.get(node1.get_name(), [])
        if (node2.get_name() in values):
            return True

        return False

    def is_required(self, node1: Node, node2: Node) -> bool:
        """
        check whether the edge node1 --> node2 is required

        Parameters
        ----------
        node1: the from node in edge which is checked
        node2: the to node in edge which is checked

        Returns
        -------
        if the  edge node1 --> node2 is required, then return True, otherwise False.
        """

        values = self.dict_required.get(node1.get_name(), [])
        if (node2.get_name() in values):
            return True

        return False