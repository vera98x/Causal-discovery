from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import filecmp
import matplotlib.pyplot as plt
from csv_to_df import retrieveDataframe
import numpy as np
from typing import Dict
from TrainRideObject import TrainRideObject
import math
import pandas as pd
from causallearn.utils.GraphUtils import GraphUtils

def gg2txt(gg : GeneralGraph, filename : str) -> None:
    nodes = gg.get_nodes()
    edges = gg.get_graph_edges()
    f = open(filename, "w")
    f.write("Graph Nodes: \n")
    for i, node in enumerate(nodes):
        node_name = node.get_name()
        if i == 0:
            f.write(node_name)
        else:
            f.write(";" + node_name)
    f.write("\n\n")
    f.write("Graph Edges:\n")
    for i, edge in enumerate(edges):
        node_name1 = edge.get_node1().get_name()
        node_name2 = edge.get_node2().get_name()
        symbol1 = "<" if edge.get_endpoint1() == Endpoint.ARROW else "-"
        symbol2 = ">" if edge.get_endpoint2() == Endpoint.ARROW else "-"
        f.write(str(i+1)+". " + node_name1 + " " + symbol1 + "-" + symbol2 + " " + node_name2 + "\n")
    f.close()

def printpfd():
    # A custom function to calculate
    # probability distribution function
    def pdf(x):
        mean = np.mean(x)
        std = np.std(x)
        y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
        return y_out

    export_name = 'Data/2019-03-01_2019-05-31.csv'  # 'Data/Ut_2022-01-01_2022-12-10_2.csv' #'Data/6100_jan_nov_2022_2.csv'
    list_of_trainseries = ['500E', '500O', '600E', '600O', '700E', '700O', '1800E', '1800O''6200E', '6200O', '8100E',
                           '8100O', '9000E', '9000O', '12600E', '76200O', '78100E', '78100O', '79000E', '79000O'
                           ]
    df, sched = retrieveDataframe(export_name, True, list_of_trainseries)
    df['delay'] = df['delay'].map(lambda x: int(x / 60))
    # To generate an array of x-values
    datacolumn = df['delay'].tolist()
    x = np.sort(np.asarray(datacolumn))

    # To generate an array of
    # y-values using corresponding x-values
    y = pdf(x)

    # Plotting the bell-shaped curve
    plt.style.use('seaborn')
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color='black',
             linestyle='dashed')

    plt.scatter(x, y, marker='o', s=25, color='red')
    plt.show()

def plotDelay():
    export_name = 'Data/2019-03-01_2019-05-31.csv'  # 'Data/Ut_2022-01-01_2022-12-10_2.csv' #'Data/6100_jan_nov_2022_2.csv'
    list_of_trainseries = ['500E', '500O', '600E', '600O', '700E', '700O', '1800E', '1800O''6200E', '6200O', '8100E',
                           '8100O', '9000E', '9000O', '12600E','76200O', '78100E', '78100O', '79000E', '79000O'
                           ]
    # extract dataframe and impute missing values
    df, sched = retrieveDataframe(export_name, True, list_of_trainseries)
    df.to_csv("mp_asn_modified.csv", index=False, sep = ";" )
    df['delay'] = df['delay'].map(lambda x: int(x/60))
    ma = df['delay'].max()
    mi = df['delay'].min()

    df.hist(column = 'delay', bins=list(range(mi, ma)))
    plt.xlim(-8, 10)
    plt.ylim(-3, 30000)
    plt.show()

def createIDTRODict(sched_with_classes : np.array) -> Dict[str, TrainRideObject]:
    '''
    Creates a dictionary for looking up the tro-schedule using the TRO-id
    :param sched_with_classes:
    :return:
    '''
    result_dict = {}
    for trn in sched_with_classes:
        result_dict[trn.getSmallerID()] = trn
    return result_dict

def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def plots():
    events = [("Bl", "8100O"), ("Bl", "8100E"), ("Hgv", "8100O"), ("Hgv", "8100E"), ("Asn", "500O"), ("Asn", "700O"),
              ("Asn", "8100O"), ("Mp", "500E"), ("Mp", "700E"), ("Mp", "8100E")]
    for event in events:
        preamble = "Data/PrimaryDelays/missing_values_as_nan_filtered_on_timestamp/"
        my_file = open(preamble + event[0] + "_" + event[1] + ".txt", "r")
        content = my_file.read()
        my_file.close()
        c = content.replace('[', '').replace(']', '').replace('\n', '').replace(" ", "")
        delays = list(map(lambda x: float(x), c.split(",")))
        delays = list(filter(lambda x: x <= 900 and x >= -300, delays))
        print(event)
        bins = math.ceil((max(delays) - min(delays)) / 60) * 3 #per 20 seconds one bin
        plt.hist(delays, bins=bins)
        plt.show()

def plots_timeseries_delay():
    df = pd.read_csv('Test17_Rotterdam/nn_input.csv', sep=";")
    group_id = df.groupby("id")
    grouped_by_id = [group_id.get_group(x) for x in group_id.groups]
    for index, group in enumerate(grouped_by_id):
        print(group.id.values[0])
        delays = group.delay.values
        x = list(range(len(delays)))
        plt.scatter(x, delays)
        plt.show()

def infoStation():
    df = pd.read_csv('Test21_add_extra_columns/nn_input.csv', sep=";")
    group_drp = df.groupby("drp")
    grouped_by_drp = [group_drp.get_group(x) for x in group_drp.groups]
    for index, group in enumerate(grouped_by_drp):
        print(group.drp.values[0])
        delays = group.delay.values
        print("min delay : ", min(delays))
        print("max delay : ", max(delays))
        print("percentage punctual (<=4 min) : ", len([i for i in delays if i<=(4*60)])/len(delays) * 100)
        print("avg delay: ",sum(delays)/len(delays))
        buffers = group.buffer.values
        print("avg buffer: ", sum(buffers) / len(buffers))
        print()

def compare_distribution_scatter():
    df = pd.read_csv('Test18_variance/nn_output_2023_03_22_10_47_47_16.34.csv', sep=";")
    group_trainserie = df.groupby("drp")
    grouped_by_trainserie = [group_trainserie.get_group(x) for x in group_trainserie.groups]
    for index, group in enumerate(grouped_by_trainserie):
        print(group.drp.values[0])
        predictions = group.prediction.values
        actual = group.actual.values
        x = list(range(len(predictions)))
        plt.scatter(x, predictions)
        plt.scatter(x, actual)
        plt.show()

def compare_distribution_histogram():
    df = pd.read_csv('Test18_variance/nn_output_2023_03_22_10_47_47_16.34.csv', sep=";")
    group_trainserie = df.groupby("drp")
    grouped_by_trainserie = [group_trainserie.get_group(x) for x in group_trainserie.groups]
    for index, group in enumerate(grouped_by_trainserie):
        print(group.drp.values[0])
        predictions = group.prediction.values
        actual = group.actual.values
        x = list(range(len(predictions)))
        bins_pred = math.ceil(abs(max(predictions) - min(predictions))/10)#per 10 seconds one bin
        bins_act = math.ceil(abs(max(actual) - min(actual))/10)  # per 10 seconds one bin
        plt.hist(actual, bins=bins_act, color="orange")
        plt.hist(predictions, bins=bins_pred, color = "blue")

        plt.show()

def traveldistance_tester(path):
    '''
    This function tests if all the neccesary distances are present within the provided dataset. If this is not provided,
    the missing connections are printed, such that it is known which need to be included manually in the file. It uses the schedule to test all next stops
    :param path:
    :return:
    '''
    filename = "Data/Dienstregelpuntverbindingen_afstanden.csv"
    df = pd.read_csv(filename, sep=";")
    df = df[["Van", "Naar", "Aantal hectometers"]]
    df["Aantal hectometers"] = df["Aantal hectometers"].astype('int')
    distance_dict = df.groupby('Van')[['Naar', 'Aantal hectometers']].apply(
        lambda x: x.set_index('Naar').to_dict(orient='index')).to_dict()

    df = pd.read_csv(path+"/sched_done.csv", sep=";")
    group_drp = df.groupby(["basic_treinnr"])
    grouped_by_drp = [group_drp.get_group(x) for x in group_drp.groups]
    for index, group in enumerate(grouped_by_drp):
        list_of_drp = group.basic_drp.values
        for i in range(1,len(list_of_drp)):
            prev = list_of_drp[i-1]
            curr = list_of_drp[i]
            if curr == prev:
                traveldistance = 0
            else:
                # on purpose no default value, should give an error when a distance is not known
                try:
                    traveldistance = distance_dict[curr][prev]["Aantal hectometers"]
                except:
                    print(curr, "and" , prev, "are not present in the dict")

def saveGraph(inputfile, destination):
    gg = txt2generalgraph(inputfile)
    pdy = GraphUtils.to_pydot(gg)
    pdy.write_png(destination)

def comparingInteraction(path):
    '''
    Compare our found interactions with the interactions found by ProRail
    :param path:
    :return:
    '''
    df_dict = {}
    gg_dict = {}

    df = pd.read_csv(path+"/nn_input.csv", sep=";")
    df = df.dropna(subset=["cause"])
    df = df[["id", "cause"]]
    for id, row in df.iterrows():
        l = df_dict.get(row[0], [])
        new_l = l.append(row[1])
        df_dict[row[0]] = new_l

    gg = txt2generalgraph(path + "/6100_FAS.txt")
    nodes = gg.get_nodes()
    for node in nodes:
        parents = gg.get_parents(node)
        if(len(parents) >0):
            parent_list = [parent.get_name() for parent in parents]
            gg_dict[node.get_name()] = parent_list

    # comparing the dicts
    #first compare how much of the found deps in gg are correct
    df = pd.DataFrame(columns=["key", "overlapping", "not_overlapping"])
    for key, value in gg_dict.items():
        df_vals = df_dict.get(key, [])
        overlapping = list(set(value).intersection(df_vals))
        not_overlapping = list(set(value) ^ set(overlapping))
        df.append([key, overlapping, not_overlapping], ignore_index=True)
    df.to_csv("gg_overlap.csv", index=False, sep=";")

    df = pd.DataFrame(columns=["key", "overlapping", "not_overlapping"])
    for key, value in df_dict.items():
        df_vals = gg_dict.get(key, [])
        overlapping = list(set(value).intersection(df_vals))
        not_overlapping = list(set(value) ^ set(overlapping))
        df.append([key, overlapping, not_overlapping], ignore_index=True)
    df.to_csv("df_overlap.csv", index=False, sep=";")

def print_number_freight(filename):
    df = pd.read_csv(filename, sep=";")
    print(len(df))
    df_freight = df.loc[df['basic_treinnr_rijkarakter'] == "GO"]
    print(len(df_freight))
    print("Percentage: ", 100/len(df)*len(df_freight))

def accuracy_per_group(filename, destination_path, list_to_group_on = None, bucket = False):
    '''
    Function is used to estimate the errors per a provided bucket to separate the result set on
    :param filename:
    :param destination_path:
    :param list_to_group_on:
    :param bucket:
    :return:
    '''
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