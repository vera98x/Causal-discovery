"""### Create SuperGraph"""
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GraphClass import CausalGraph
import numpy as np
import time
from enum import Enum
from FastBackgroundKnowledge import FastBackgroundKnowledge
from typing import Dict, Tuple, List
from TrainRideNode import TrainRideNode
import datetime

class Graph_type(Enum):
    SUPER = 1
    MINIMAL = 2
    SWITCHES = 3

def createStationDict(train_serie_day: np.array) -> Dict[str, Tuple[datetime.time, TrainRideNode]]:
    # Partition the data per station, such that we can find dependencies within a station fast
    station_dict = {}  # example: bkl: [(8:15, TrainRideNode1), (13:05, TrainRideNode2)]

    # for each station, list all the trains that arrive there with its arrival time and TrainRideNode
    for trn in train_serie_day:
        if trn.getStation() not in station_dict.keys():
            station_dict[trn.getStation()] = [(trn.getPlannedTime(), trn)]
        else:
            arrivalpairs = station_dict[trn.getStation()]
            station_dict[trn.getStation()] = arrivalpairs + [(trn.getPlannedTime(), trn)]
    return station_dict

class DomainKnowledge:
    def __init__(self, sched_with_classes : np.array, type : Graph_type):
        self.graph_type = type
        self.sched_with_classes = sched_with_classes
        self.column_names = np.array(list(map(lambda x: x.getSmallerID(), sched_with_classes)))
        self.station_dict = createStationDict(sched_with_classes)
    def makeEverythingForbidden(self, train_serie_day: np.array, bk: FastBackgroundKnowledge) -> FastBackgroundKnowledge:
        '''Makes every node combination marked as forbidden'''
        train_serie_day_names = [trn.getSmallerID() for trn in train_serie_day]
        for trn_index in range(len(train_serie_day_names)):
            trn_name = train_serie_day_names[trn_index]
            if trn_index != len(train_serie_day_names) -1:
                other_deps = train_serie_day_names[:trn_index] + train_serie_day_names[trn_index+1:]
            else:
                other_deps = train_serie_day_names[:trn_index]
            bk.addForbiddenDependency_dict(trn_name, other_deps)
        return bk
    def addRequiredBasedTrainSerie(self, train_serie_day: np.array, bk: FastBackgroundKnowledge) -> FastBackgroundKnowledge:
        '''Marks the trains with the same train number at consequtive events required'''
        # trainseries contains a 1D array containing TreinRideNode data of one day
        # for each train ride and every stop/station, add a chain of dependencies
        # s1->s2->s3
        prev_name = ""
        prev_trainNumber = ""
        for trn in train_serie_day:
            # skip first station
            if trn.getTrainRideNumber() != prev_trainNumber:
                prev_trainNumber = trn.getTrainRideNumber()
                prev_name = trn.getSmallerID()
                continue
            bk.addDependency_dict(prev_name, trn.getSmallerID())
            prev_name = trn.getSmallerID()

        return bk

    def addPossibleBasedStation(self, bk: FastBackgroundKnowledge) -> FastBackgroundKnowledge:
        '''For trains within 15 minutes and at the same station, remove the forbidden dependency'''
        buffer = 15  # minutes
        for station, station_list in self.station_dict.items():
            # sort on planned time
            station_list.sort(key=lambda x: x[0])
            for station_index in range(len(station_list)):
                (time_trn, trn) = station_list[station_index]
                for station2_index in range(station_index+1, len(station_list)):
                    (other_time_trn, other_trn) = station_list[station2_index]
                    if (other_time_trn - time_trn).total_seconds() <= (buffer * 60): # TODO: why not return when the time is out of range?
                        if(self.graph_type == Graph_type.SUPER):
                            bk.addDependency_dict(trn.getSmallerID(), other_trn.getSmallerID())
                        if (self.graph_type == Graph_type.SWITCHES):
                            shared = any(x in trn.wissels for x in other_trn.wissels)
                            if shared:
                                bk.addDependency_dict(trn.getSmallerID(), other_trn.getSmallerID())
                            else:
                                bk.removeForbiddenDependency_dict(trn.getSmallerID(), other_trn.getSmallerID())
                                bk.removeForbiddenDependency_dict(other_trn.getSmallerID(), trn.getSmallerID())
                        if(self.graph_type == Graph_type.MINIMAL):
                            bk.removeForbiddenDependency_dict(trn.getSmallerID(), other_trn.getSmallerID())
                            bk.removeForbiddenDependency_dict(other_trn.getSmallerID(), trn.getSmallerID())
        return bk
    def create_background_knowledge(self) -> FastBackgroundKnowledge:
        # important: the order of train rides should be ordered by train numbers and planned type

        # create background knowledge, take into account:
        # 1. The train that follows its route (chain of actions), depends on the previous action of the train, thus there is a direct cause. (required edge)
        # 2. Trains that are on the same station within 15 minutes may have a direct cause. (possible edge)
        # 3. Trains that are not in the same station cannot have a direct cause. (forbidden edge)

        bk = FastBackgroundKnowledge()
        # first add the required edges, then the forbidden edges (forbidden edges checks if some edge was already required, then it does not add a forbidden edge)
        print("Make everything forbidden")
        bk = self.makeEverythingForbidden(self.sched_with_classes, bk)
        print("Required based on train serie")
        bk = self.addRequiredBasedTrainSerie(self.sched_with_classes, bk) # TODO: why provide a parameter that is a "self." that is already accessible
        print("add required based on stations")
        bk = self.addPossibleBasedStation(bk)
        return bk


    def create_background_knowledge_wrapper(self) -> FastBackgroundKnowledge:

        print("Creating background knowledge")
        start = time.time()
        background = self.create_background_knowledge()
        end = time.time()
        print("creating schedule took", end - start, "seconds")
        return background