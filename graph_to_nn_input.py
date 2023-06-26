import copy
from datetime import date, datetime, time
from causallearn.graph.GeneralGraph import GeneralGraph
from PC_and_background import PCAndBackground
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Utils import createIDTRODict, nan_equal

class Station_type(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

class Weekday(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4

class Peak(Enum):
    NONPEAK = 0
    MORNINGPEAK = 1
    EVENINGPEAK = 2

global station_dict
filename = "Data/Ontkoppelpuntenlijst.csv"
df = pd.read_csv(filename, sep=";")
df = df[["Dienstregelpunt", "Logistieke functionaliteit voor VGMI database"]]
df = df.rename(columns={"Dienstregelpunt": "drp", "Logistieke functionaliteit voor VGMI database": "type"})
station_dict = df.set_index('drp').T.to_dict("list")

global minimal_dwell_time
minimal_dwell_time = {"Asn" : {"SPR" : 60, "IC" : 60, "ICE": np.nan},
                      "Hgv" : {"SPR" : 42, "IC" : 54, "ICE": np.nan},
                      "Bl" : {"SPR" : 42, "IC" : 54, "ICE": np.nan},
                      "Mp" : {"SPR" : 42, "IC" : 54, "ICE": np.nan},
                      # rtd - ddr
                      "Rtd" : {"SPR" : 90, "IC" : 120, "THA": 240},
                      "Wiltn" : {"SPR" : 42, "IC" : 54, "ICE": np.nan},
                      "Rtb": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Wiltz": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Rtz": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Rtst": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Rlb": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Brd": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Kfhaz": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Zwd": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Grbr": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ddr" : {"SPR" : 60, "IC" : 90, "ICE": np.nan},
                      # Asd - Ut
                      "Asd": {"SPR": 120, "IC": 180, "THA": 300, "ICE": np.nan},
                      "Ods": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Dgrw": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Asdma": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Asdm": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Asa": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Dvd": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Asb": {"SPR": 60, "IC": 60, "ICE": np.nan},
                      "Dvaz": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ashd": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ac": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Aco": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Bkl": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Bkla": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Mas": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Utzl": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Utma": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ut": {"SPR": 90, "IC": 120, "ICE": 120},
                      # Ut (not included) - Ehv
                      "Utvr": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Utza": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Utln": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Htn": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Htnc": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Lek": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Cl": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Gdm": {"SPR": 60, "IC": 60, "ICE": np.nan},
                      "Mbtwan": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Mbtwaz": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Zbm": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ozbm": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Hdl": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ht": {"SPR": 90, "IC": 120, "ICE": np.nan},
                      "Vga": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Vg": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Btl": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Lpe": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Beto": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Bet": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "At": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ehs": {"SPR": 42, "IC": 54, "ICE": np.nan},
                      "Ehv": {"SPR": 90, "IC": 120, "ICE": np.nan}
}


global distance_dict
filename = "Data/Dienstregelpuntverbindingen_afstanden.csv"
df = pd.read_csv(filename, sep=";")
df = df[["Van", "Naar", "Aantal hectometers"]]
df["Aantal hectometers"] = df["Aantal hectometers"].astype('int')
distance_dict = df.groupby('Van')[['Naar','Aantal hectometers']].apply(lambda x: x.set_index('Naar').to_dict(orient='index')).to_dict()

class NN_input_row:
    def __init__(self, trn, prev_event_trn, row_info, time_interval):
        global station_dict
        global minimal_dwell_time
        global distance_dict
        self.label = trn.getSmallerID()
        self.drp = trn.getStation()
        self.date = trn.date #row_info[0]
        self.trn_delay = row_info[0]
        self.dep_delay = row_info[1:] # consists of 6 items prev_event, prev_prev_event, prev_platform and 3 other delays
        self.timeinterval = time_interval
        # Todo: read this directly from data
        self.type_train = trn.getTraintype() #"SPR" if(trn.getTrainSerie() == "8100E" or trn.getTrainSerie() == "8100O") else "IC"
        self.train_mat = trn.getTrainmat()
        self.type_station = station_dict[trn.getStation()]
        self.day_of_the_week = self.getDay(self.date)
        self.peakhour = self.getPeak(trn.getPlannedTime_time())
        self.buffer = trn.getBuffer()
        self.traveltime = trn.getTraveltime()
        self.trainserie = trn.getTrainSerie()
        self.activity = trn.getActivity()
        self.minimal_dwell_time = minimal_dwell_time[trn.getStation()].get(self.type_train, np.nan)
        self.speed = trn.getSpeed()
        self.slack = trn.getSlack()
        self.cause = trn.getCause()
        if prev_event_trn == None:
            self.traveldistance = np.nan
        elif trn.getStation() == prev_event_trn.getStation():
            self.traveldistance = 0
        else:
            # on purpose no default value, should give an error when a distance is not known
            self.traveldistance = distance_dict[trn.getStation()][prev_event_trn.getStation()]["Aantal hectometers"]

    def __eq__(self, other):
        a = np.array(self.dep_delay)
        b = np.array(other.dep_delay)
        return (self.label == other.label and self.drp == other.drp and self.date == other.date and ((self.trn_delay == other.trn_delay) or (np.isnan(self.trn_delay) and np.isnan(other.trn_delay))) and nan_equal(a,b) and self.timeinterval == other.timeinterval)

    def __repr__(self):
        return (str(self.label) + "_+" + str(self.drp) + "_+" + str(self.date) + "_+" + str(self.trn_delay) + "_+" + str(self.dep_delay) + "_+" + str(self.timeinterval))

    def getDay(self, trn_date : datetime.date) -> Weekday:
        day = trn_date.weekday()
        if day == 0:
            return Weekday.MONDAY
        if day == 1:
            return Weekday.TUESDAY
        if day == 2:
            return Weekday.WEDNESDAY
        if day == 3:
            return Weekday.THURSDAY
        else:
            return Weekday.FRIDAY
    def getPeak(self, t : datetime.time) -> Peak:
        # note, this program only uses weekdays. In weekends it is always non-peak
        interval_peak = [(time(6,30,0), time(9,0,0)), (time(16,00,0), time(18,00,0))]
        if t >= interval_peak[0][0] and t < interval_peak[0][1]:
            return Peak.MORNINGPEAK
        if t >= interval_peak[1][0] and t < interval_peak[1][1]:
            return Peak.EVENINGPEAK
        return Peak.NONPEAK

class NodeAndParents:
    def __init__(self, trn):
        self.trn = trn
        self.prev_event = None
        self.prev_prev_event = None
        self.dep0_platform = None
        self.dep = [] # dependencies

    def get_itself_and_all_parents(self, max_len_dep):
        dep = (self.dep + [None]*max_len_dep)[:max_len_dep]  # get the list of dependencies and add if necessary 3 None
        l = [self.trn, self.prev_event, self.prev_prev_event, self.dep0_platform] + dep
        return l


class NN_samples:
    def __init__(self, gg: GeneralGraph, sched_with_classes: np.array, df):
        self.nodes_and_parents_list = []
        self.gg = gg
        self.sched_with_classes = sched_with_classes
        self.id_trn = createIDTRODict(sched_with_classes)
        self.df = df
        self.list_input_rows = []

    def convert_graph_to_class(self) -> None:
        '''
        updates the sample_data list with Dependency_nodes
        :param events:
        :return None:
        '''
        # get all nodes from the graph
        nodes = self.gg.get_nodes()
        # fidn for each node its dependencies
        for node in nodes:
            # retrieve the corresponding train, This is the train from the schedule
            trn = self.id_trn[node.get_name()]
            # initialize a dependency node
            node_trn = NodeAndParents(trn)
            # get the parents of the node
            parents = self.gg.get_parents(node)
            for parent in parents:
                # the parent trn
                p_trn = self.id_trn[parent.get_name()]
                # if parent and trn have the same train number, assign it to the prev event
                if (trn.getTrainRideNumber() == p_trn.getTrainRideNumber()):
                    node_trn.prev_event = p_trn
                    # get parentparent
                    parentparents = self.gg.get_parents(parent)
                    parentparent_trns = [self.id_trn[parentparent.get_name()] for parentparent in parentparents]
                    parentparent = [x for x in parentparent_trns if x.getTrainRideNumber() == p_trn.getTrainRideNumber() ]
                    if len(parentparent) > 0:
                        node_trn.prev_prev_event = parentparent[0]
                    else:
                        node_trn.prev_prev_event = None
                # if the parent is in front of the next train at the same platform
                elif (trn.getPlatform() == p_trn.getPlatform() and trn.getStation() == p_trn.getStation()):
                    # if this variable is already filled
                    if(node_trn.dep0_platform != None):
                        # check which train was really the last train at that point, the other train is then a "general" dependency
                        if(node_trn.dep0_platform.getPlannedTime() < p_trn.getPlannedTime()):
                            node_trn.dep.append(node_trn.dep0_platform)
                            node_trn.dep0_platform = p_trn
                            #print("platform Update", p_trn.getID())
                        else:
                            node_trn.dep.append(p_trn)
                    # if dep0_platform is not filled, fill it
                    else:
                        node_trn.dep0_platform = p_trn
                # the rest are general dependencies
                else:
                    node_trn.dep.append(p_trn)
            dep_list = node_trn.dep
            sorted_dep_list = sorted(dep_list, key=lambda x: x.getPlannedTime(), reverse=False)
            node_trn.dep = sorted_dep_list
            self.nodes_and_parents_list.append(node_trn)

        return None

    def findDelaysFromData(self, trn_matrix : np.array) -> None:
        self.list_input_rows = np.empty([len(self.nodes_and_parents_list) * len(trn_matrix)], dtype=NN_input_row)
        all_delays = np.array([[x.getDelay() for x in trn_matrix_row] for trn_matrix_row in trn_matrix])
        #dates = np.array([[x.getDate() for x in trn_matrix_row] for trn_matrix_row in trn_matrix])[:,0]
        max_len_dep = max([len(x.dep) for x in self.nodes_and_parents_list])

        row_index = 0
        for index, node in enumerate(self.nodes_and_parents_list):
            trn_index = node.trn.getIndex()
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
            current_time = np.array([node.trn.getPlannedTime()] * len(timings))
            time_interval = list(map(lambda x: x.total_seconds(), (current_time - timings)))[
                            -4:]  # only take the platform dep and the other deps

            for i, row in enumerate(resultmatrix):
                self.list_input_rows[row_index] = (NN_input_row(trn = trn_matrix[i][trn_index], prev_event_trn = node.prev_event, row_info = [*row], time_interval = time_interval))
                row_index += 1
    def NN_input_rows_to_df(self, filename):
        x = []
        y = []
        label_encoder = LabelEncoder()
        # do day of the week to categorical
        integers_day_of_the_week = [x.day_of_the_week.value for x in self.list_input_rows]
        encoded_day_of_the_week = to_categorical(integers_day_of_the_week)
        # do type train to categorical
        labels_type_train = [x.type_station for x in self.list_input_rows]
        integer_type_train = label_encoder.fit_transform(labels_type_train)
        encoded_type_train = to_categorical(integer_type_train)
        # do train_serie to categorical
        #integers_trainserie = [x.trainserie for x in self.list_nn_input]
        #encoded_trainserie = to_categorical(integers_trainserie)
        # do station_type to categorical
        labels_station_type = [x.type_station for x in self.list_input_rows]
        integer_station_type = label_encoder.fit_transform(labels_station_type)
        encoded_station_type = to_categorical(integer_station_type)
        # do peakhour to categorical
        integers_peakhour = [x.peakhour.value for x in self.list_input_rows]
        encoded_peakhour = to_categorical(integers_peakhour)
        # do activity to categorical (sinds activity is a string, an extra conversion step is added)
        labels_activity = [x.activity for x in self.list_input_rows]
        integer_activity = label_encoder.fit_transform(labels_activity)
        encoded_activity = to_categorical(integer_activity)
        # do train_mat to categorical (since train_mat is a string, an extra conversion step is added)
        labels_train_mat = [x.train_mat for x in self.list_input_rows]
        integer_train_mat = label_encoder.fit_transform(labels_train_mat)
        encoded_train_mat = to_categorical(integer_train_mat)

        for index, item in enumerate(self.list_input_rows):
            row = [item.label, item.drp, item.trainserie, item.activity, *item.dep_delay, *item.timeinterval, *encoded_type_train[index], *encoded_station_type[index],
                   *encoded_day_of_the_week[index], *encoded_peakhour[index], item.buffer, item.traveltime, *encoded_activity[index], item.minimal_dwell_time, item.traveldistance,*encoded_train_mat[index], item.speed, item.slack, item.cause]
            x.append(row)
            y.append(item.trn_delay)

        #create dataframe to save intermediate computation time next time
        column_names = ["id", "drp", "trainserie", "act", "prev_event", "prev_prev_event","dep0_platform", *["dep"+str(i) for i in range(1,len(self.list_input_rows[0].dep_delay)+1-3)], *["timeinterval_" + str(i) for i in range(len(self.list_input_rows[0].timeinterval))] , *["traintype_" + str(i) for i in range(len(encoded_type_train[0]))], *["type_station_" + str(i) for i in range(len(encoded_station_type[0]))],
                        *["day_of_the_week_" + str(i) for i in range(len(encoded_day_of_the_week[0]))], *["peakhour_" + str(i) for i in range(len(encoded_peakhour[0]))], "buffer", "traveltime", *["activity_" + str(i) for i in range(len(encoded_activity[0]))],
                        "minimal_dwell_time", "traveldistance", *["trainmat_" + str(i) for i in range(len(encoded_train_mat[0]))], "speed", "slack", "cause"]

        df = pd.DataFrame(x, columns=column_names)
        # maybe add travel time between each stop?
        df = df.assign(delay = y)
        df.to_csv(filename, index = False, sep = ";")
        return