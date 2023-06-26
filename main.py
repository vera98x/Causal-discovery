from df_to_trn import TRN_matrix_to_delay_matrix, dfToTrainRides, createindexDict
from csv_to_df import retrieveDataframe
from createBackgroundKnowledge import DomainKnowledge, Graph_type
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from PC_and_background import PCAndBackground
from graph_to_nn_input import NN_samples
from NeuralNetwork import preTrainedNN, precisionNN, testPretrainedNN, test_precision
from Utils import gg2txt, traveltime_tester, saveGraph, print_number_freight
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
    delay_matrix = TRN_matrix_to_delay_matrix(trn_matrix)

    sched_with_classes = dfToTrainRides(sched)[0]
    print("Amount of variables: ", len(trn_matrix[0]))
    column_names = np.array(list(map(lambda x: x.getSmallerID(), sched_with_classes)))

    # create a background and its schedule (background for Pc or FCI, cg_sched for GES)
    dk = DomainKnowledge(sched_with_classes, 'Test14_testsamples/sched.png', Graph_type.SWITCHES)
    bk, cg_sched = dk.create_background_knowledge_wrapper()  # get_CG_and_background(smaller_dataset, 'Results/sched.png')

    gg_fas = cg_sched.G
    sample_changer = NN_samples(gg_fas, sched_with_classes, df)
    sample_changer.convert_graph_to_class()
    sample_changer.findDelaysFromData(trn_matrix)
    sample_changer.NN_input_rows_to_df("Test14_testsamples/nn_input.csv")

    #sample_changer.NN_input_class_to_matrix("Test14_testsamples/nn_input.csv")

def main(calculate_background : bool, path, import_name):
    print("Phase: dataframe cleaning and to TRO")
    # extract dataframe and impute missing values
    df, sched = retrieveDataframe(import_name, True)
    df.to_csv(path+"/df_done.csv", index=False, sep=";")
    sched.to_csv(path+"/sched_done.csv", index=False, sep=";")
    print("done extracting", len(df))
    tro_schedule_list = dfToTrainRides(sched)[0]
    # create schedule of the Train ride nodes
    index_dict = createindexDict(tro_schedule_list)
    # change the dataframe to trainRideNodes
    tro_matrix = dfToTrainRides(df, index_dict)

    print("translating dataset to 2d array for algo")
    print("Amount of variables: ", len(tro_matrix[0]))

    # translate the TrainRideNodes to delays
    column_names = np.array(list(map(lambda x: x.getSmallerID(), tro_schedule_list)))
    delays_matrix = TRN_matrix_to_delay_matrix(tro_matrix)

    if(calculate_background):
        # create a background from the schedule
        dk = DomainKnowledge(tro_schedule_list, Graph_type.MINIMAL)
        bk = dk.create_background_knowledge_wrapper()

        hybrid_method = PCAndBackground('mv_fisherz', delays_matrix,
                                     path +'/causal_graph.png', tro_schedule_list, bk, column_names)
        gg = hybrid_method.apply_pc_with_background(False)
        gg2txt(gg, path+"/causal_graph.txt")
    else:
        print("retrieve value from file")
        start = time.time()
        gg = txt2generalgraph(path+"/causal_graph.txt")
        end = time.time()
        print("Retrieving graph from file took", end - start, "seconds")

    # -----------------------

    print("gg to nn")
    start = time.time()
    sample_changer = NN_samples(gg, tro_schedule_list, df)
    sample_changer.convert_graph_to_class() #
    print("find delays from data")
    sample_changer.findDelaysFromData(tro_matrix)
    end = time.time()
    print("creating new input took", end - start, "seconds")
    sample_changer.NN_input_rows_to_df(path + "/nn_input.csv")


def splitPrimaryDelay(df):
    dependency_delays = df[["prev_event","prev_prev_event","prev_platform","dep1","dep2","dep3"]]
    dependency_delays.mask(dependency_delays <=0, 0, inplace=True)
    dependency_delays = dependency_delays.fillna(0)
    dependency_delays["total"] = dependency_delays["prev_event"] + dependency_delays["prev_prev_event"]+ dependency_delays["prev_platform"] +\
                                dependency_delays["dep1"] + dependency_delays["dep2"] + dependency_delays["dep3"]
    prim = df[dependency_delays['total']>0]
    no_delay = df[dependency_delays['total']<=0]
    return prim, no_delay

def variances(df = None):
    if df is None:
        df = pd.read_csv('Test17_Rotterdam/nn_input.csv', sep=";")
    #only take id and delay, everything else is not important
    #df = df[["id","delay"]]
    df = df.dropna(subset=["delay"])
    group_id = df.groupby(['trainserie','drp'])
    grouped_by_id = [group_id.get_group(x) for x in group_id.groups]
    for index, group in enumerate(grouped_by_id):
        prim_df, no_delay_df = splitPrimaryDelay(group)
        print(group.trainserie.values[0], group.drp.values[0])
        #group = group.query('delay <= 900 & delay >= -900')
        print("Prim: ------------------------------------")
        prim_df = prim_df.assign(variance=prim_df.delay.values.var())
        prim_df = prim_df.assign(std=prim_df.delay.values.std())
        prim_df = prim_df.assign(mean=prim_df.delay.values.mean())
        print("amount of samples: ", len(prim_df.trainserie.values))
        print("Variance :", prim_df.delay.values.var())
        print("std: ", prim_df.delay.values.std())
        print("No delay: ------------------------------------")
        no_delay_df = no_delay_df.assign(variance=no_delay_df.delay.values.var())
        no_delay_df = no_delay_df.assign(std=no_delay_df.delay.values.std())
        no_delay_df = no_delay_df.assign(mean=no_delay_df.delay.values.mean())
        print("amount of samples: ", len(no_delay_df.trainserie.values))
        print("Variance :", no_delay_df.delay.values.var())
        print("std: ", no_delay_df.delay.values.std())
        print()
        grouped_by_id[index] = pd.concat([prim_df, no_delay_df])
    df_new = pd.concat(grouped_by_id)
    return df_new

def accuracy_per_group(filename, destination_path, list_to_group_on = None, bucket = False):
    df = pd.read_csv(filename, sep=";")
    if bucket:
        df["bucket"] = df["actual"].apply(lambda x: math.floor(x // 30))
    else:
        df["bucket"] = 0
    print(list_to_group_on)
    if ( list_to_group_on != None):
        group_drp = df.groupby(list_to_group_on)
        grouped_by_drp = [group_drp.get_group(x) for x in group_drp.groups]
    else:
        grouped_by_drp = [df]
    df = pd.DataFrame(columns=["Index","Amount of samples","15_sec", "30_sec", "MAE", "RMSE", "prev_median", "prev_mean", "MdAPE", "MAPE"])
    for index, group in enumerate(grouped_by_drp):
        print(group.trainserie.values[0], group.drp.values[0], "bucket:", group[[*list_to_group_on]].values[0])
        f = lambda x: float(x)
        g = lambda s: s.replace(",", ".")
        y = np.array(list(map(f, group.actual.values)))
        y_hat_str = np.array(list(map(f, group.prediction.values)))
        y_hat = np.array(list(map(f, y_hat_str)))
        prev = np.array(list(map(f, group.prev.values)))
        diff = abs(y_hat - y)
        print("amount of test samples: ", len(group.prediction.values))
        percentage15 = sum(i <= 15 for i in diff)/len(diff)
        percentage30 = sum(i <= 30 for i in diff) / len(diff)
        print("Percentage within 15 sec: ", percentage15)
        print("Percentage within 30 sec: ", percentage30)
        mae = sum(abs(y_hat - y))/len(y)
        rmse = math.sqrt(sum(abs(y_hat - y)**2) / len(y))
        prev_median = np.median(np.nan_to_num((abs(y_hat - y)/abs(prev-y) * 100)))
        mdape = np.median(np.nan_to_num(abs((y_hat - y)/y)*100))
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("Comparing to prev, median: ", prev_median)
        print("MdAPE, median: ", mdape)

        indexes_mdape = np.nonzero( ((y < 1) & (y > -1)))
        y_filtered_mdape = np.delete(y, indexes_mdape)
        y_hat_filtered_mdape = np.delete(y_hat, indexes_mdape)
        mape = str(round(np.mean(np.nan_to_num(abs((y_hat_filtered_mdape - y_filtered_mdape) / y_filtered_mdape) * 100)),3))

        print("MdAPE, mean: ", mape)

        diff_prev = prev-y
        indexes_prev = np.nonzero( ((diff_prev < 1) & (diff_prev > -1)))
        diff_prev_filtered = np.delete(diff_prev, indexes_prev)
        y_filtered_prev = np.delete(y, indexes_prev)
        y_hat_filtered_prev = np.delete(y_hat, indexes_prev)
        prev_mean = str(round(np.mean(np.nan_to_num((abs(y_hat_filtered_prev - y_filtered_prev)/abs(diff_prev_filtered) * 100))),3))
        print("Comparing to prev, mean: ",  prev_mean)

        new_row = [group[[*list_to_group_on]].values[0], len(group.prediction.values), round(percentage15,3), round(percentage30,3), round(mae,3), round(rmse,3), round(prev_median,3), prev_mean, round(mdape,3),mape]
        df = df.append(pd.Series(new_row, index=df.columns), ignore_index = True)
        print()

    df.to_csv(destination_path+"/acc_per_group.csv", index = False, sep = ";")

def relativeError():
    df = pd.read_csv('Test21_add_extra_columns/nn_output_2023_04_06_15_18_16_11.12.csv', sep=";")
    prev_error = abs(df.prev.values - df.actual.values)
    estimation_error = abs(df.prediction.values - df.actual.values)
    relative_error = estimation_error/prev_error
    print(np.sort(relative_error))
    print("Median: ", np.median(relative_error))
    print("Mean: ", np.mean(relative_error))


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

def preprocess_df(df, weights, subset_with_only_interaction, dep_columns):
    # if the prev event or the delay is not known, remove it from the dataframe
    df = df.dropna(subset=['delay'])
    df = df.dropna(subset=['prev_event'])
    df = df.dropna(subset=["traveltime"])

    print(len(df))
    if weights:
        # df["weigth"] = np.where(((df['prev_platform'] > 60) | (df['dep1'] > 60) | (df['dep2'] > 60) | (df['dep3'] > 60)) & (df['delay'] > 60), 1000, 1)
        df["delay_jump"] = abs(df["prev_event"] - df["delay"])
        df["weight"] = np.where((df["delay_jump"] >= 30), df["delay_jump"]*1.5, 1)
        #df["weight"] = np.where(~df["cause"].isna(), 50, 1)
        df = df.drop(columns=["delay_jump"])
    else:
        df["weight"] = 1
    if subset_with_only_interaction:
        df = df.dropna(subset=["dep1", "dep0_platform"], how="all")
        df = df.dropna(subset=['cause'])
    print("After: ", len(df))
    df = df.fillna(-50)
    # add all columns except id and delay as input
    if (not dep_columns):
        df = df[df.columns.drop(list(df.filter(regex='dep')))]
        df = df[df.columns.drop(list(df.filter(regex='timeinterval_')))]
        df = df[df.columns.drop(list(df.filter(regex='prev_platform')))]
    df = df[df.columns.drop(list(df.filter(regex='trainmat_')))]
    return df

def main_nn(train_pre_nn: bool, test_pre_trained_nn: bool, train_precision_nn: bool, test_precision_bool :bool, subset_with_only_interaction: bool, weights :bool, dep_columns : bool, path, experiment_name1 ='default', experiment_name2 ='default', temp = ""):
    df_train = pd.read_csv(path+'/nn_input_train.csv', sep = ";")
    df_test = pd.read_csv(path + '/nn_input_test.csv', sep=";")
    print("train: ", len(df_train), "test: ", len(df_test))
    df_train = preprocess_df(df_train, weights, subset_with_only_interaction, dep_columns)
    df_test = preprocess_df(df_test, weights, subset_with_only_interaction, dep_columns)

    remaining_df_train = df_train[df_train.columns[~df_train.columns.isin(["id", 'delay', "trainserie", "drp", "act", "weight", "cause", "index_total"])]]
    remaining_df_test= df_test[df_test.columns[~df_test.columns.isin(["id", 'delay', "trainserie", "drp", "act", "weight", "cause", "index_total"])]]


    print(remaining_df_train.columns)
    x_train, x_test = remaining_df_train.values.tolist(), remaining_df_test.values.tolist()
    y_train, y_test = df_train["delay"].tolist(), df_test["delay"].tolist()
    w_train, w_test = df_train["weight"].tolist(), df_test["weight"].tolist()
    trainserie_train, trainserie_test = df_train.trainserie.values, df_test.trainserie.values
    drp_train, drp_test = df_train.drp.values, df_test.drp.values
    prev_train, prev_test = df_train.prev_event.values, df_test.prev_event.values
    if "act" not in df_train.columns:
        df_train["act"] = "not_included"
        df_test["act"] = "not_included"
    act_train,  act_test= df_train.act.values, df_test.act.values
    index_train, index_test = df_train.index_total.values, df_test.index_total.values
    # first use the general nn to train
    if train_pre_nn:
        preTrainedNN((x_train,x_test), (y_train, y_test),(w_train,w_test), (trainserie_train,trainserie_test), (drp_train,drp_test), (prev_train,prev_test), (act_train,act_test), (index_train, index_test), path, str(remaining_df_train.columns), experiment_name1)
    if test_pre_trained_nn:

        total_df = testPretrainedNN(x_test, y_test, w_test, trainserie_test, drp_test, prev_test, act_test, index_test, path + "/Models",
                                    str(remaining_df_test.columns),
                                    experiment_name2)
        total_df.to_csv(path + "/tested_pre_trained_merged_files.csv", index=False, sep=";")
    if train_precision_nn:
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
            precisionNN(x_train,y_train,w_train,trainserie_train, drp_train, prev_train, act_train, path+"/Models", str(remaining_df_train.columns), experiment_name2)

        #total_df.to_csv(path + "/merged_files.csv", index=False, sep=";")

    if test_precision_bool:
        # then use a second NN for more precision
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
                total_df = test_precision(x_test, y_test, w_test, trainserie_test, drp_test, prev_test, act_test, index_test, path + "/Models", str(remaining_df_test.columns),
                                       experiment_name2)
            else:
                new_df = test_precision(x_test, y_test, w_test, trainserie_test, drp_test, prev_test, act_test, index_test, path + "/Models", str(remaining_df_test.columns),
                                     experiment_name2)
                total_df = pd.concat([total_df, new_df])
        total_df.to_csv(path + "/tested_merged_files_" + temp + ".csv", index=False, sep=";")

# main(calculate_background = True, path="Test50_asd", datasetname = "Data/Asd-Ut.csv" )
# create_train_and_test(path_input="Test50_asd", path_output="Test50_asd")
# main_nn(train_pre_nn= True, test_pre_trained_nn= False, train_precision_nn=True, test_precision_bool = True, subset_with_only_interaction= False, weights= False, dep_columns= True,
#        path="Test50_asd", experiment_name1="Ehv_dep_train_buffer_15_min", experiment_name2="Asd_dep_test_buffer_15_min", temp="column_causes")