
from causallearn.graph.GeneralGraph import GeneralGraph
from typing import List, Union
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Utils import createIDTRODict
from NN_input_row import NN_input_row
from TrainRideObject import TrainRideObject

class NodeAndParents:
    def __init__(self, tro):
        self.tro = tro
        self.prev_event = None
        self.prev_prev_event = None
        self.dep0_platform = None
        self.dep = [] # dependencies

    def get_itself_and_all_parents(self, max_len_dep : int) -> List[Union[TrainRideObject,None]]:
        '''
        Return all the variables of the class, in a specific order. The number of dependencies is determined by max_len_dep. If the length is shorter than that, add some None values.
        :param max_len_dep:
        :return List[Union[TrainRideObject,None]]:
        '''
        dep = (self.dep + [None]*max_len_dep)[:max_len_dep]  # get the list of dependencies and add if necessary some None
        l = [self.tro, self.prev_event, self.prev_prev_event, self.dep0_platform] + dep
        return l


class NN_samples:
    def __init__(self, general_graph: GeneralGraph, sched_with_classes: np.array):
        self.nodes_and_parents_list : List[NodeAndParents] = []
        self.gg = general_graph
        self.id_tro = createIDTRODict(sched_with_classes)
        self.input_rows_list : List[NN_input_row]= []

    def convert_graph_to_nodes_and_parents_list(self) -> None:
        '''
        For each node in the graph, it finds and classifies its parents and creates a NodeAndParents class, which is added to the nodes_and_parents_list
        :param None:
        :return None:
        '''
        # get all nodes from the graph
        nodes = self.gg.get_nodes()
        # find for each node its dependencies
        for node in nodes:
            # retrieve the corresponding train, This is the train from the schedule
            tro = self.id_tro[node.get_name()]
            # initialize a dependency node
            node_tro = NodeAndParents(tro)
            # get the parents of the node
            parents = self.gg.get_parents(node)
            for parent in parents:
                # the parent tro
                parent_tro = self.id_tro[parent.get_name()]
                # if parent and tro have the same train number, assign it to the prev event
                if (tro.getTrainRideNumber() == parent_tro.getTrainRideNumber()):
                    node_tro.prev_event = parent_tro
                    # get parentparent
                    parentparents = self.gg.get_parents(parent)
                    parentparent_tros = [self.id_tro[parentparent.get_name()] for parentparent in parentparents]
                    parentparent = [x for x in parentparent_tros if x.getTrainRideNumber() == parent_tro.getTrainRideNumber() ]
                    if len(parentparent) > 0:
                        node_tro.prev_prev_event = parentparent[0]
                    else:
                        node_tro.prev_prev_event = None
                # if the parent is in front of the next train at the same platform
                elif (tro.getPlatform() == parent_tro.getPlatform() and tro.getStation() == parent_tro.getStation()):
                    # if this variable is already filled
                    if(node_tro.dep0_platform != None):
                        # check which train was really the last train at that point, the other train is then a "general" dependency
                        if(node_tro.dep0_platform.getPlannedTime() < parent_tro.getPlannedTime()):
                            node_tro.dep.append(node_tro.dep0_platform)
                            node_tro.dep0_platform = parent_tro
                            #print("platform Update", parent_tro.getID())
                        else:
                            node_tro.dep.append(parent_tro)
                    # if dep0_platform is not filled, fill it
                    else:
                        node_tro.dep0_platform = parent_tro
                # the rest are general dependencies
                else:
                    node_tro.dep.append(parent_tro)
            dep_list = node_tro.dep
            sorted_dep_list = sorted(dep_list, key=lambda x: x.getPlannedTime(), reverse=False)
            node_tro.dep = sorted_dep_list
            self.nodes_and_parents_list.append(node_tro)

        return None

    def nodes_and_parents_list_to_input_rows_list(self, tro_matrix : np.array) -> None:
        '''
        for each node in the nodes_and_parents_list, find the delays of the node and its parents and provide this to the NN_input_row class. Each NN_input_row is added to the input_rows_list
        :param tro_matrix:
        :return None:
        '''
        self.input_rows_list = np.empty([len(self.nodes_and_parents_list) * len(tro_matrix)], dtype=NN_input_row)
        all_delays = np.array([[x.getDelay() for x in tro_matrix_row] for tro_matrix_row in tro_matrix])
        max_len_dep = max([len(x.dep) for x in self.nodes_and_parents_list])

        row_index = 0
        for index, node in enumerate(self.nodes_and_parents_list):
            tro_index = node.tro.getIndex()
            # get the index of the train nodes which are dependencies for the current node
            indexes_of_dependencies = [x.getIndex() if x != None else None for x in node.get_itself_and_all_parents(max_len_dep)]

            # get the destination index for all the missing values
            nonexisting_future_indexes_dependencies = [i for i, x in enumerate(indexes_of_dependencies) if x == None]
            # get the destination index for all the present values
            existing_future_indexes_dependencies = [i for i, x in enumerate(indexes_of_dependencies) if x != None]

            # get the value index which can be retrieved in the 'all_delays' matrix
            lookup_existing_indexes_dependencies = [x for i, x in enumerate(indexes_of_dependencies) if x != None]
            # get all the values of the dependencies according to the given indexes
            all_delays_filtered = all_delays[:,lookup_existing_indexes_dependencies]
            # first create an empty resultmatrix
            resultmatrix = np.empty([len(all_delays), len(indexes_of_dependencies)])

            # fill with the found dependencies its delays
            resultmatrix[:, existing_future_indexes_dependencies] = np.array(all_delays_filtered)
            # fill all other parts with np.nan
            resultmatrix[:, nonexisting_future_indexes_dependencies] = np.nan

            # include the timings, for which they are per node the same, so declare per loop only once and always add the same
            timings = np.array([x.getPlannedTime() if x != None else pd.Timestamp('NaT').to_pydatetime() for x in
                                node.get_itself_and_all_parents(max_len_dep)])
            current_time = np.array([node.tro.getPlannedTime()] * len(timings))
            time_interval = list(map(lambda x: x.total_seconds(), (current_time - timings)))[
                            -4:]  # only take the platform dep and the other deps

            for i, row in enumerate(resultmatrix):
                self.input_rows_list[row_index] = (NN_input_row(tro= tro_matrix[i][tro_index], prev_event_tro= node.prev_event, row_info = [*row], time_interval = time_interval))
                row_index += 1
    def input_rows_list_to_df(self, filename :str) -> None:
        '''
        For each item in the input_rows_list, add this as a row in the dataframe. The complete dataframe is saved using the filename variable
        :param filename:
        :return:
        '''
        x = []
        y = []
        label_encoder = LabelEncoder()
        # do day of the week to categorical
        integers_day_of_the_week = [x.day_of_the_week.value for x in self.input_rows_list]
        encoded_day_of_the_week = to_categorical(integers_day_of_the_week)
        # do type train to categorical
        labels_type_train = [x.type_station for x in self.input_rows_list]
        integer_type_train = label_encoder.fit_transform(labels_type_train)
        encoded_type_train = to_categorical(integer_type_train)
        # do station_type to categorical
        labels_station_type = [x.type_station for x in self.input_rows_list]
        integer_station_type = label_encoder.fit_transform(labels_station_type)
        encoded_station_type = to_categorical(integer_station_type)
        # do peakhour to categorical
        integers_peakhour = [x.peakhour.value for x in self.input_rows_list]
        encoded_peakhour = to_categorical(integers_peakhour)
        # do activity to categorical (sinds activity is a string, an extra conversion step is added)
        labels_activity = [x.activity for x in self.input_rows_list]
        integer_activity = label_encoder.fit_transform(labels_activity)
        encoded_activity = to_categorical(integer_activity)

        for index, item in enumerate(self.input_rows_list):
            row = [item.label, item.drp, item.trainserie, item.activity, *item.dep_delay, *item.timeinterval, *encoded_type_train[index], *encoded_station_type[index],
                   *encoded_day_of_the_week[index], *encoded_peakhour[index], item.buffer, item.traveltime, *encoded_activity[index], item.minimal_dwell_time, item.traveldistance, item.speed, item.slack, item.cause]
            x.append(row)
            y.append(item.tro_delay)

        #create dataframe to save intermediate computation time next time
        column_names = ["id", "drp", "trainserie", "act", "prev_event", "prev_prev_event","dep0_platform", *["dep" + str(i) for i in range(1, len(self.input_rows_list[0].dep_delay) + 1 - 3)], *["timeinterval_" + str(i) for i in range(len(self.input_rows_list[0].timeinterval))] , *["traintype_" + str(i) for i in range(len(encoded_type_train[0]))], *["type_station_" + str(i) for i in range(len(encoded_station_type[0]))],
                        *["day_of_the_week_" + str(i) for i in range(len(encoded_day_of_the_week[0]))], *["peakhour_" + str(i) for i in range(len(encoded_peakhour[0]))], "buffer", "traveltime", *["activity_" + str(i) for i in range(len(encoded_activity[0]))],
                        "minimal_dwell_time", "traveldistance", "speed", "slack", "cause"]

        df = pd.DataFrame(x, columns=column_names)
        # maybe add travel time between each stop?
        df = df.assign(delay = y)
        df.to_csv(filename, index = False, sep = ";")
        return


    def graph_to_nn_input(self, tro_matrix : np.array, filename : str) -> None:
        '''
        Converts the graph and the tro_matrix to a csv-file that can be used as input for the Neural Network
        :param tro_matrix:
        :param filename:
        :return:
        '''
        self.convert_graph_to_nodes_and_parents_list()
        self.nodes_and_parents_list_to_input_rows_list(tro_matrix)
        self.input_rows_list_to_df(filename)
        return None