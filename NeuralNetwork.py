
# -----IMPORTANT FOR REPRODUCABILITY
seed_value = 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)
#--------------------------------------------------------
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import math
import matplotlib.pyplot as plt
import mlflow.keras

mlflow.keras.autolog()

def mape_(y_true, y_pred):
    return abs((y_pred - y_true)/y_true) * 100

def variance_laplace(y_true, y_pred, weights):
    return abs(y_pred - y_true)/weights*100

def weights_mae(y_true, y_pred, weights):
    return abs(y_pred - y_true) * weights

def preTrainedNN(x_raw : Tuple[List[List[float]], List[List[float]]], y_raw : Tuple[List[float], List[float]], w_raw : Tuple[List[float], List[float]], trainserie : Tuple[List[str], List[str]], drp : Tuple[List[str], List[str]], prev : Tuple[List[str], List[str]], act :Tuple[List[str], List[str]], index: Tuple[List[int], List[int]], path : str, df_columns, experiment_name):
    mlflow.set_experiment(experiment_name)
    input_dim = len(x_raw[0][0])
    # x_train_raw, x_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw, trainserie_train, trainserie_test, drp_train, drp_test, prev_train, prev_test, \
    #     act_train, act_test = train_test_split(x_raw, y_raw, weight, trainserie, drp, prev, act, test_size=0.20, random_state=42)
    x_train = tf.constant(x_raw[0])
    y_train = tf.constant(y_raw[0])
    w_train = tf.constant(w_raw[0])
    x_test = tf.constant(x_raw[1])
    y_test = tf.constant(y_raw[1])
    w_test = tf.constant(w_raw[1])

    def rmse_():
        return tf.sqrt(tf.reduce_mean((out - true)**2))

    def mse_():
        return abs(out - true)**2

    def mae_():
        return abs(out - true)

    print(x_train.shape, y_train.shape)
    inp = tf.keras.layers.Input(shape=(input_dim,))
    true = tf.keras.layers.Input((1,))
    weights = tf.keras.layers.Input((1,))
    h1 = tf.keras.layers.Dense(256, activation="ReLU", kernel_initializer="RandomUniform")(inp)
    out = tf.keras.layers.Dense(1, activation="PReLU", kernel_initializer= tf.keras.initializers.GlorotNormal(seed = 42) )(h1)

    model = tf.keras.Model([inp, true, weights], out)
    model.add_loss(weights_mae(true, out, weights))
    model.add_metric(mse_(), name = "mse_", aggregation="mean")
    model.add_metric(mae_(), name="mae_", aggregation="mean")
    model.add_metric(rmse_(), name = "rmse_", aggregation="mean")
    model.compile(optimizer='adam',
                  loss=None,
                  )
    with mlflow.start_run(run_name="nn_model"):
        model_history = model.fit(x=[x_train, y_train, w_train],y=None, epochs=20, batch_size=128, validation_split=0.2)
        mlflow.log_text(df_columns, "columns.txt")
        mlflow.tensorflow.log_model(model=model, artifact_path="model")
    mlflow.end_run()

    print("eval train")
    print(model.evaluate([x_train, y_train, w_train], None, verbose=2))
    print("eval test")

    len_samples = len(y_raw[1])
    print(model.evaluate([x_test,  y_test, w_test], None, verbose=2))
    print("_______________")
    prediction = model.predict([x_test,  y_test, w_test])
    prediction_flatten = [item for sublist in prediction for item in sublist]
    y_test_list = y_test.numpy().tolist()
    comparison = list(zip(prediction_flatten, y_test_list))
    mae = sum([abs(x - y) for x, y in comparison]) / len(prediction_flatten)
    filename1 = path + "/nn_output" + "_" + str(
        round(mae, 2)) + "_" + str(datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))

    showMetrixes(prediction, y_test, trainserie[1], drp[1], prev[1], act[1], filename1, df_columns, True, index[1])
    #model.save_weights(path + '/firstModelWeights')
    model.save(path+'/Models/firstModel')

    #plothistory(model_history)

def precisionNN(x_train_raw : List[List[float]], y_train_raw : List[float], w_train_raw : List[float], trainserie : List[str], drp : List[str], prev: List[str], act : List[str], path, df_columns : str, experiment_name):
    mlflow.set_experiment(experiment_name)

    # x_train_raw, x_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw, trainserie_train, trainserie_test, drp_train, drp_test, prev_train, prev_test, act_train, act_test = train_test_split(
    #     x_raw, y_raw, w_raw, trainserie, drp, prev, act, test_size=0.20, random_state=42)

    x_train = tf.constant(x_train_raw)
    y_train = tf.constant(y_train_raw)

    # x_test = tf.constant(x_test_raw)
    # y_test = tf.constant(y_test_raw)

    w_train = tf.constant(w_train_raw)
    # w_test = tf.constant(w_test_raw)

    savedModel = tf.keras.models.load_model(path+'/firstModel')
    # savedModel = mlflow.keras.load_model(
    #     model_uri=f"models:/first_model/{3}"
    # )
    #savedModel = savedModel.load_weights(path+'firstModelWeights')

    print("------------------------------------ New training")
    with mlflow.start_run(run_name=str(drp[0]) + "_" + str(trainserie[0])):
        model_history = savedModel.fit(x = [x_train, y_train, w_train], y = None, epochs=10, batch_size=128, validation_split=0.2)
        mlflow.log_text(df_columns, "columns.txt")
        mlflow.tensorflow.log_model(model=savedModel, artifact_path="model")
    mlflow.end_run()
    # print("_______________")
    # prediction = savedModel.predict([x_test, y_test, w_test])
    #
    # prediction_flatten = [item for sublist in prediction for item in sublist]
    #
    # y_test_list = y_test.numpy().tolist()
    # comparison = list(zip(prediction_flatten, y_test_list))
    # mae = sum([abs(x - y) for x, y in comparison]) / len(prediction_flatten)
    filename1 = path + "/" + str(drp[0]) + "_" + str(trainserie[0])
    savedModel.save(filename1)
    # return showMetrixes(prediction, y_test, trainserie_test, drp_test, prev_test, act_test, filename1, "", False)

def test_precision(x_test_raw : List[List[float]], y_test_raw: List[float], weight_test_raw: List[float], trainserie: List[str], drp: List[str], prev:List[str], act: List[str], index: List[int], path: str, df_columns, experiment_name):
    mlflow.set_experiment(experiment_name)
    x_test = tf.constant(x_test_raw)
    y_test = tf.constant(y_test_raw)
    w_test = tf.constant(weight_test_raw)

    savedModel = tf.keras.models.load_model(path + '/' + str(drp[0]) + "_" + str(trainserie[0]))
    print("eval test")
    with mlflow.start_run(run_name=str(drp[0]) + "_" + str(trainserie[0])):
        model_result = savedModel.evaluate([x_test, y_test, w_test], None, verbose=2)
        print(model_result)
        mlflow.log_metric("mae_", model_result[2])
        mlflow.log_metric("rmse_", model_result[3])
        mlflow.log_metric("mse_", model_result[1])
        mlflow.log_text(df_columns, "columns.txt")
    mlflow.end_run()

    prediction = savedModel.predict([x_test, y_test, w_test])

    prediction_flatten = [item for sublist in prediction for item in sublist]

    y_test_list = y_test.numpy().tolist()
    comparison = list(zip(prediction_flatten, y_test_list))
    mae = sum([abs(x - y) for x, y in comparison]) / len(prediction_flatten)
    filename1 = path + "/" + str(
        round(mae, 2)) + "_testing_" + str(datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
    return showMetrixes(prediction, y_test, trainserie, drp, prev, act, filename1, "", False, index)

def testPretrainedNN(x_raw : List[List[float]], y_raw : List[float], weight : List[float], trainserie : List[str], drp : List[str], prev :List[str], act : List[str], index : List[int], path : str, df_columns, experiment_name):
    mlflow.set_experiment(experiment_name)
    x = tf.constant(x_raw)
    y = tf.constant(y_raw)
    w= tf.constant(weight)
    # x_train_raw, x_test_raw, y_train_raw, y_test_raw, w_train_raw, w_test_raw, trainserie_train, trainserie_test, drp_train, drp_test, prev_train, prev_test, act_train, act_test = train_test_split(
    #     x_raw, y_raw, weight, trainserie, drp, prev, act, test_size=0.20, random_state=42)

    # x_train = tf.constant(x_train_raw)
    # y_train = tf.constant(y_train_raw)
    #
    # x_test = tf.constant(x_test_raw)
    # y_test = tf.constant(y_test_raw)
    #
    # w_train = tf.constant(w_train_raw)
    # w_test = tf.constant(w_test_raw)

    savedModel = tf.keras.models.load_model(path + '/firstModel')
    print("eval test")
    with mlflow.start_run(run_name="testing_nn_model"):
        model_result = savedModel.evaluate([x, y, w], None, verbose=2)
        print(model_result)
        mlflow.log_metric("mae_", model_result[2])
        mlflow.log_metric("rmse_", model_result[3])
        mlflow.log_metric("mse_", model_result[1])
        mlflow.log_text(df_columns, "columns.txt")
    mlflow.end_run()

    prediction = savedModel.predict([x,  y, w])

    prediction_flatten = [item for sublist in prediction for item in sublist]

    y_test_list = y.numpy().tolist()
    comparison = list(zip(prediction_flatten, y_test_list))
    mae = sum([abs(x - y) for x, y in comparison]) / len(prediction_flatten)
    filename1 = path + "/" + str(
        round(mae, 2)) + "_testing_" + str(datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
    return showMetrixes(prediction, y, trainserie, drp, prev, act, filename1, "", False, index)

def showMetrixes(prediction, y_test, trainserie_test, drp_test, prev_test, act_test, filename : str, df_columns :str, to_file: bool, index: List[int] = []):
    prediction_flatten = [item for sublist in prediction for item in sublist]

    y_test_list = y_test.numpy().tolist()
    comparison = list(zip(prediction_flatten, y_test_list))
    mae = sum([abs(x - y) for x, y in comparison]) / len(prediction_flatten)
    print("mean average error:", mae)
    print("squared mean error:", sum([abs(x - y) ** 2 for x, y in comparison]) / len(prediction_flatten))
    print("root squared mean error:",
          math.sqrt(sum([abs(x - y) ** 2 for x, y in comparison]) / len(prediction_flatten)))
    print("within 15 sec: ", sum([abs(x - y) <= 15 for x, y in comparison]) / len(prediction_flatten))
    print("within 30 sec: ", sum([abs(x - y) <= 30 for x, y in comparison]) / len(prediction_flatten))

    df = pd.DataFrame()
    df["prediction"] = prediction_flatten
    df["actual"] = y_test_list
    df["difference"] = df["prediction"] - df["actual"]
    df["trainserie"] = trainserie_test
    df["drp"] = drp_test
    df["prev"] = prev_test
    df["act"] = act_test
    if len(index) != 0:
        df["index"] = index
    df = df.round(4)
    if (to_file):
        filename_csv = filename + ".csv"
        df.to_csv(filename_csv, index=False, sep=";")
        f = open(filename + ".txt", "w")
        f.write("--------- Configuration -------- \n")
        f.write("input, 256 tanh RandomUniform, 1 prelu glorotNormal (seed = 42) \n")
        f.write("optimizer = adam, epochs=80, batch_size=128 \n")
        f.write("--------- train --------\n")
        f.write("mean average error: \n")
        f.write("squared mean error: \n")
        f.write("--------- test -------- \n")
        f.write("mean average error: " + str(sum([abs(x - y) for x, y in comparison]) / len(prediction_flatten)) + "\n")
        f.write(
            "squared mean error: " + str(sum([abs(x - y) ** 2 for x, y in comparison]) / len(prediction_flatten)) + "\n")
        f.write("root squared mean error:" + str(
            math.sqrt(sum([abs(x - y) ** 2 for x, y in comparison]) / len(prediction_flatten))) + "\n")
        f.write("within 15 sec: " + str(sum([abs(x - y) <= 15 for x, y in comparison]) / len(prediction_flatten)) + "\n")
        f.write("within 30 sec: " + str(sum([abs(x - y) <= 30 for x, y in comparison]) / len(prediction_flatten)) + "\n")
        f.write(df_columns + "\n")
        f.close()
    return df