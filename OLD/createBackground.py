"""### Create BackgroundKnowlegde"""
import numpy as np
from typing import Dict, Tuple, List
from TrainRideObject import TrainRideObject
import datetime

def createStationDict(train_serie_day: np.array) -> Dict[str, Tuple[datetime.time, TrainRideObject]]:
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

def variableNamesToNumber(day : List[TrainRideObject]) -> Tuple[Dict[str,str], Dict[str,str]]:
    counter = 1
    trn_name_id_dict = {}
    id_trn_name_dict = {}
    for trn in day:
        trn_name_id_dict[trn.getID()] = 'X' + str(counter)
        id_trn_name_dict['X' + str(counter)] = trn.getID()
        counter += 1

    return trn_name_id_dict, id_trn_name_dict