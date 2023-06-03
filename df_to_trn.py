# clean data
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from TrainRideNode import TrainRideNode

def createindexDict(trn_sched_matrix : List[TrainRideNode]):
    dict = {}
    for trn in trn_sched_matrix:
        index = trn.getIndex()
        name = trn.getSmallerID()
        dict[name] = index
    return dict


def dfToTrainRides(df : pd.DataFrame, index_dict = None) -> np.array:
    gb = df.groupby(['date'])

    data = [gb.get_group(x) for x in gb.groups]
    column_len = len(data[0])

    dataset_with_classes_new = np.empty((len(data), column_len)).astype(TrainRideNode)
    for index_x, day_df in enumerate(data):
        for index_y, trainride in day_df.iterrows():

            node = TrainRideNode(trainSerie = trainride['basic_treinnr_treinserie'], trainRideNumber = trainride['basic_treinnr'], stationName = trainride['basic_drp'], platformNumber = trainride['basic_spoor_simpel'],
                                 activity = trainride['basic_drp_act'], delay = trainride['delay'],  plannedTime = trainride["basic_plan"], globalTime= trainride["global_plan"], buffer = trainride["buffer"], date = trainride["date"],
                                 traveltime = trainride["traveltime"], wissels = trainride["wissels"], speed = trainride["speed"], traintype = trainride["basic_treinnr_rijkarakter"], trainmat = trainride["vklbammat_soorttype_uitgevoerd"], slack=trainride["slack"], cause_train = trainride["stipt_oorzaak_treinr"], index = None)
            if index_dict == None:
                i_y = index_y % column_len
            else:
                i_y = index_dict[node.getSmallerID()]
            node.setIndex(i_y)
            dataset_with_classes_new[index_x, i_y] = node

    for j in range(len(dataset_with_classes_new[0])):
        column_ = dataset_with_classes_new[:,j]
        names_ = np.array(list(map(lambda x: x.getSmallerID(), column_)))
        occurences = np.count_nonzero(names_ == names_[0])
        if(occurences < len(data)):
            print(names_[0], " is not unique!!!!!, there are ", occurences, "present")
            print(names_)
    return dataset_with_classes_new

# Create format for data to feed in the causal discovery algorithms
def TRN_matrix_to_delay_matrix(dataset_with_classes : np.array) -> np.array:
    array_with_delays_2d_new = np.zeros((len(dataset_with_classes), len(dataset_with_classes[0]))).astype(float)
    for index, day in enumerate(dataset_with_classes):
        array_with_delays_2d_new[index] = np.array(list(map(lambda x: x.getDelay(), day)))

    return array_with_delays_2d_new