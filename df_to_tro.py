# clean data
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from TrainRideObject import TrainRideObject

def create_dict_TROid_column_index(tro_schedule_list : List[TrainRideObject]) -> Dict[str, int]:
    dict = {}
    for tro in tro_schedule_list:
        index = tro.getIndex()
        name = tro.getSmallerID()
        dict[name] = index
    return dict


def dfToTrainRides(df : pd.DataFrame, column_index_dict = None) -> np.array:
    ''' The dataframe is changed to a matrix with each row denoting one date and each column represents an event (train 4024 arriving at Bkl for example)'''
    # important: the dataframe needs to be sorted, so to be sure, this is sorted again
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic_treinnr", "basic_plan"])
    df = df.reset_index(drop=True)

    gb = df.groupby(['date'])

    list_of_different_days = [gb.get_group(x) for x in gb.groups]
    column_len = len(list_of_different_days[0])

    tro_matrix = np.empty((len(list_of_different_days), column_len)).astype(TrainRideObject)
    for index_x, day_df in enumerate(list_of_different_days):
        for index_y, trainride in day_df.iterrows():
            tro = TrainRideObject(trainSerie = trainride['basic_treinnr_treinserie'], trainRideNumber = trainride['basic_treinnr'], stationName = trainride['basic_drp'], platformNumber = trainride['basic_spoor_simpel'],
                                   activity = trainride['basic_drp_act'], delay = trainride['delay'], plannedTime = trainride["basic_plan"], globalTime= trainride["global_plan"], buffer = trainride["buffer"], date = trainride["date"],
                                   traveltime = trainride["traveltime"], wissels = trainride["wissels"], speed = trainride["speed"], traintype = trainride["basic_treinnr_rijkarakter"], trainmat = trainride["vklbammat_soorttype_uitgevoerd"], slack=trainride["slack"], cause_train = trainride["stipt_oorzaak_treinr"], index = None)
            if column_index_dict == None:
                #index_y keeps increasing per day, so by applying the modulo, the "normal" column index is captured
                i_y = index_y % column_len
            else:
                # to be sure that per day the same tro is at the same column index, we retrieve the planned column index by the dictionary
                i_y = column_index_dict[tro.getSmallerID()]
            # add the additional information for the column index at the TRO
            tro.setColumnIndex(i_y)
            # place the TRO at this specific column
            tro_matrix[index_x, i_y] = tro

    return tro_matrix

def TRO_matrix_to_delay_matrix(tro_matrix : np.array) -> np.array:
    '''This function translate the np.array filled with TRO to a np.array of ints denoting the delay of each TRO
    This format is created for data to feed in the causal discovery algorithms'''
    array_with_delays_2d_new = np.zeros((len(tro_matrix), len(tro_matrix[0]))).astype(float)
    for index, day in enumerate(tro_matrix):
        array_with_delays_2d_new[index] = np.array(list(map(lambda x: x.getDelay(), day)))

    return array_with_delays_2d_new