from df_to_tro import TRO_matrix_to_delay_matrix, dfToTrainRides, create_dict_TROid_column_index
from csv_to_df import retrieveDataframe
from createBackgroundKnowledge import DomainKnowledge, Graph_type
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from PC_and_background import PCAndBackground
from graph_to_nn_input import NN_samples
from NeuralNetwork import train_pre_trained_nn, train_fine_tuned_nn, test_pre_trained_nn, test_fine_tuned_nn
from Utils import gg2txt, traveltime_tester, saveGraph, print_number_freight, accuracy_per_group
import numpy as np
import pandas as pd
import math
import time


def main_test():
    df, sched = retrieveDataframe("Test14_testsamples/df_test_speed_slack_cause.csv", True)
    df.to_csv("Test14_testsamples/df_done.csv", index=False, sep=";")
    sched.to_csv("Test14_testsamples/sched_done.csv", index=False, sep=";")
    trn_matrix = dfToTrainRides(df)
    # # translate the TrainRideNodes to delays
    delay_matrix = TRO_matrix_to_delay_matrix(trn_matrix)

    sched_with_classes = dfToTrainRides(sched)[0]
    print("Amount of variables: ", len(trn_matrix[0]))
    column_names = np.array(list(map(lambda x: x.getSmallerID(), sched_with_classes)))

    # create a background and its schedule (background for Pc or FCI, cg_sched for GES)
    dk = DomainKnowledge(delay_matrix, 'Test14_testsamples/sched.png', Graph_type.SWITCHES)
    bk, cg_sched = dk.create_background_knowledge_with_timing()  # get_CG_and_background(smaller_dataset, 'Results/sched.png')

    gg_fas = cg_sched.G
    sample_changer = NN_samples(gg_fas, sched_with_classes, df)
    sample_changer.convert_graph_to_nodes_and_parents_list()
    sample_changer.nodes_and_parents_list_to_input_rows_list(trn_matrix)
    sample_changer.input_rows_list_to_df("Test14_testsamples/nn_input.csv")

    #sample_changer.NN_input_class_to_matrix("Test14_testsamples/nn_input.csv")

def create_nn_input_from_df(calculate_background : bool, path, dataset_filename):
    print("Phase: dataframe cleaning and converting to TRO")
    #---------------------------------------------------- start retrieving and cleaning dataframe
    # extract dataframe and impute missing values
    df, sched = retrieveDataframe(dataset_filename, True)
    df.to_csv(path+"/df_done.csv", index=False, sep=";")
    sched.to_csv(path+"/sched_done.csv", index=False, sep=";")
    print("done cleaning df, number of rows:", len(df))
    # ---------------------------------------------------- end retrieving and cleaning dataframe
    # ---------------------------------------------------- start translating df to TrainRideObjects
    # create schedule of the Train ride nodes
    tro_schedule_list = dfToTrainRides(df=sched)[0]
    #create a dictionary providing the column indexes per tro, to make sure that all the same event end up in the same column
    column_index_dict = create_dict_TROid_column_index(tro_schedule_list=tro_schedule_list)
    # change the dataframe to trainRideNodes
    tro_matrix = dfToTrainRides(df=df, column_index_dict=column_index_dict)
    # translate the TrainRideNodes to a matrix of delays
    delays_matrix = TRO_matrix_to_delay_matrix(tro_matrix=tro_matrix)
    print("done translating df to TRO")
    # ---------------------------------------------------- end translating df to TrainRideObjects

    if(calculate_background):
        # create background knowledge from the schedule
        dk = DomainKnowledge(tro_schedule_list=tro_schedule_list, type=Graph_type.MINIMAL)
        bk = dk.create_background_knowledge_with_timing()
        #apply the pc-algorithm in combination with background knowledge
        hybrid_method = PCAndBackground(method = 'mv_fisherz', data=delays_matrix, tro_schedule_list = tro_schedule_list, bk=bk)
        general_graph = hybrid_method.apply_pc_with_background(graph_save_png_bool= False, filename=path + '/causal_graph.png')
        # save the calculated graph
        gg2txt(general_graph, path+"/causal_graph.txt")
    else:
        print("retrieve the general graph from file")
        start = time.time()
        general_graph = txt2generalgraph(path+"/causal_graph.txt")
        end = time.time()
        print("Retrieving graph from file took", end - start, "seconds")

    # -----------------------

    print("general_graph to nn inout")
    start = time.time()
    #initialize the class
    sample_changer = NN_samples(general_graph, tro_schedule_list)
    # start the conversion and provide the tro_matrix for the delays and external variables
    sample_changer.graph_to_nn_input(tro_matrix=tro_matrix, filename=path + "/nn_input.csv")
    end = time.time()
    print("creating new input took", end - start, "seconds")



def create_train_and_test(path_input, path_output):
    df = pd.read_csv(path_input + '/nn_input.csv', sep=";")
    df = df.dropna(subset=['delay'])
    df = df.dropna(subset=['prev_event'])
    df = df.dropna(subset=["traveltime"])
    # add index number
    df = df.reset_index()
    df.rename(columns={'index': 'index_total'}, inplace=True)
    train_set = df.sample(frac=0.8, random_state=42)
    test_set = df.drop(train_set.index)
    train_set.to_csv(path_output + "/nn_input_train.csv", index=False, sep=";")
    test_set.to_csv(path_output + "/nn_input_test.csv", index=False, sep=";")

def preprocess_df_nn(df, weights, subset_with_only_interaction, dep_columns):
    # if the prev event traveltime or the delay is not known, remove it from the dataframe
    df = df.dropna(subset=['delay'])
    df = df.dropna(subset=['prev_event'])
    df = df.dropna(subset=["traveltime"])

    print(len(df))
    if weights:
        #add a weight to the variables, based on their delay jump
        df["delay_jump"] = abs(df["prev_event"] - df["delay"])
        df["weight"] = np.where((df["delay_jump"] >= 30), df["delay_jump"]*1.5, 1)
        df = df.drop(columns=["delay_jump"])
    else:
        df["weight"] = 1
    if subset_with_only_interaction:
        # only keep the variables where the dependencies are non-empty and there is a cause
        df = df.dropna(subset=["dep1", "dep0_platform"], how="all")
        df = df.dropna(subset=['cause'])
    print("After: ", len(df))
    # fill the empty values with -50
    df = df.fillna(-50)
    # add all columns except id and delay as input
    if (not dep_columns):
        df = df[df.columns.drop(list(df.filter(regex='dep')))]
        df = df[df.columns.drop(list(df.filter(regex='timeinterval_')))]
        df = df[df.columns.drop(list(df.filter(regex='prev_platform')))]
    return df

def train_or_test_nn(train_pre_trained_nn_bool: bool, test_pre_trained_nn_bool: bool, train_fine_tuned_nn_bool: bool, test_fine_tuned_nn_bool :bool, subset_with_only_interaction: bool, weights :bool, dep_columns : bool, path, experiment_name1 ='default', experiment_name2 ='default', additional_file_name =""):
    # the filenames should be nn_imput_train and nn_input_test, to have a clear distinguishment between both sets.
    df_train = pd.read_csv(path+'/nn_input_train.csv', sep = ";")
    df_test = pd.read_csv(path + '/nn_input_test.csv', sep=";")
    print("train: ", len(df_train), "test: ", len(df_test))
    df_train = preprocess_df_nn(df_train, weights, subset_with_only_interaction, dep_columns)
    df_test = preprocess_df_nn(df_test, weights, subset_with_only_interaction, dep_columns)

    remaining_df_train = df_train[df_train.columns[~df_train.columns.isin(["id", 'delay', "trainserie", "drp", "act", "weight", "cause", "index_total"])]]
    remaining_df_test= df_test[df_test.columns[~df_test.columns.isin(["id", 'delay', "trainserie", "drp", "act", "weight", "cause", "index_total"])]]

    print(remaining_df_train.columns)
    x_train, x_test = remaining_df_train.values.tolist(), remaining_df_test.values.tolist()
    y_train, y_test = df_train["delay"].tolist(), df_test["delay"].tolist()
    w_train, w_test = df_train["weight"].tolist(), df_test["weight"].tolist()
    trainserie_train, trainserie_test = df_train.trainserie.values, df_test.trainserie.values
    drp_train, drp_test = df_train.drp.values, df_test.drp.values
    prev_train, prev_test = df_train.prev_event.values, df_test.prev_event.values
    act_train,  act_test= df_train.act.values, df_test.act.values
    index_train, index_test = df_train.index_total.values, df_test.index_total.values
    # first use the general nn to train
    if train_pre_trained_nn_bool:
        train_pre_trained_nn((x_train, x_test), (y_train, y_test), (w_train, w_test), (trainserie_train, trainserie_test), (drp_train, drp_test), (prev_train, prev_test), (act_train, act_test), (index_train, index_test), path, str(remaining_df_train.columns), experiment_name1)
    if test_pre_trained_nn_bool:

        total_df = test_pre_trained_nn(x_test, y_test, w_test, trainserie_test, drp_test, prev_test, act_test, index_test, path + "/Models",
                                            str(remaining_df_test.columns),
                                            experiment_name2)
        total_df.to_csv(path + "/tested_pre_trained_merged_files.csv", index=False, sep=";")
    if train_fine_tuned_nn_bool:
        # then use a second NN for more precision
        group_id = df_train.groupby(['trainserie', 'drp'])
        grouped_by_id = [group_id.get_group(x) for x in group_id.groups]
        for index, group in enumerate(grouped_by_id):
            y_train = group["delay"].tolist()
            w_train = [1] * len(group["delay"].tolist())
            # add all columns except id and delay as input
            to_include_x = group[group.columns[~group.columns.isin(["id", 'delay', "trainserie", "drp", "act", "weight", "cause", "index_total"])]]
            x_train = to_include_x.values.tolist()
            trainserie_train = group.trainserie.values
            drp_train = group.drp.values
            prev_train = group.prev_event.values
            act_train = group.act.values
            train_fine_tuned_nn(x_train, y_train, w_train, trainserie_train, drp_train, prev_train, act_train, path + "/Models", str(remaining_df_train.columns), experiment_name2)


    if test_fine_tuned_nn_bool:
        # then use a second NN for more precision, grouped by trainserie and drp
        group_id = df_test.groupby(['trainserie', 'drp'])
        grouped_by_id = [group_id.get_group(x) for x in group_id.groups]
        for index, group in enumerate(grouped_by_id):
            y_test = group["delay"].tolist()
            w_test = [1] * len(group["delay"].tolist())
            # add all columns except id and delay as input
            to_include_x = group[group.columns[~group.columns.isin(["id", 'delay', "trainserie", "drp", "act", "weight", "cause", "index_total"])]]
            x_test = to_include_x.values.tolist()
            trainserie_test = group.trainserie.values
            drp_test = group.drp.values
            prev_test = group.prev_event.values
            act_test = group.act.values
            index_test = group.index_total.values
            if (index == 0):
                total_df = test_fine_tuned_nn(x_test, y_test, w_test, trainserie_test, drp_test, prev_test, act_test, index_test, path + "/Models", str(remaining_df_test.columns),
                                                   experiment_name2)
            else:
                new_df = test_fine_tuned_nn(x_test, y_test, w_test, trainserie_test, drp_test, prev_test, act_test, index_test, path + "/Models", str(remaining_df_test.columns),
                                                 experiment_name2)
                total_df = pd.concat([total_df, new_df])
        total_df.to_csv(path + "/tested_merged_files_" + additional_file_name + ".csv", index=False, sep=";")


create_nn_input_from_df(calculate_background = True, path="Tests", dataset_filename="Data/Rtd-Ddr.csv")
# create_train_and_test(path_input="Test50_asd", path_output="Test50_asd")
# main_nn(train_pre_nn= True, test_pre_trained_nn= False, train_precision_nn=True, test_precision_bool = True, subset_with_only_interaction= False, weights= False, dep_columns= True,
#        path="Test50_asd", experiment_name1="Ehv_dep_train_buffer_15_min", experiment_name2="Asd_dep_test_buffer_15_min", temp="column_causes")