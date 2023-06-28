import copy

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import datetime as dt
from typing import List, Tuple
import math
import random

def keepTrainseries(df_input : pd.DataFrame, act_val : List[str]) ->  pd.DataFrame:
    df_input = df_input[df_input['basic_treinnr_treinserie'].isin(act_val)]

    return df_input
def keepWorkDays(df_input : pd.DataFrame) ->  pd.DataFrame:
    # translate date to number 1 - 7 and is ma - su
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    # weekdays are between 1-5
    df_input = df_input[(df_input['daynumber'] <= 5)]
    # drop the column again
    df_input = df_input.drop(columns=['daynumber'])
    return df_input
def keepWeekendDays(df_input : pd.DataFrame) ->  pd.DataFrame:
    # translate date to number 1 - 7 and is ma -su
    df_input['daynumber'] = df_input['date'].apply(lambda x: int(x.strftime("%u")))
    # weekdays are between 1-5
    df_input = df_input[(df_input['daynumber'] >= 6)]
    # drop the column again
    df_input = df_input.drop(columns=['daynumber'])
    return df_input

def makeDataUniform(df : pd.DataFrame, sched : pd.DataFrame) ->  pd.DataFrame:
    gb = df.groupby(['date'])
    grouped_by_date = [gb.get_group(x) for x in gb.groups]
    # get first dataframe as example to compare from
    sched_date = sched.iloc[0]['date']


    print("amount of days", len(grouped_by_date))
    # loop through every other frame, compare the columns
    for day_index in range(len(grouped_by_date)):
        day = grouped_by_date[day_index]
        day_date = day.iloc[0]['date']
        # find differences, and drop all overlapping values between the schedule and the day. If there are remaining
        # values, these need to be inserted or removed
        diff = pd.concat([sched, day]).drop_duplicates(subset=['basic_treinnr', 'basic_drp', 'basic_drp_act'],
                                                         keep=False)
        if (len(diff) != 0):

            # remove the extra activities
            # find activities that are not in the schedule
            remove_extra_activities = diff[~(diff["date"] == sched_date)]
            # add those activities again to the day dataframe, and drop the duplicates, such that all items that should
            # be removed are removed
            df_res = pd.concat([day, remove_extra_activities]).drop_duplicates(
                subset=['basic_treinnr', 'basic_drp', 'basic_drp_act', 'date'],
                keep=False)

            # add missing values
            add_extra_activities = diff[(diff["date"] == sched_date)]
            add_extra_activities = add_extra_activities.assign(date=day_date)
            add_extra_activities = add_extra_activities.assign(time = add_extra_activities["basic_plan"].dt.time)
            add_extra_activities.loc[:, "basic_plan"] = pd.to_datetime(add_extra_activities.date.astype(str) + ' ' + add_extra_activities.time.astype(str))
            # -------------------------------------------------------------------------------------------------- fix date jump in basic plan
            add_extra_activities["basic_plan_tomorrow"] = add_extra_activities["basic_plan"].apply(lambda x: x + dt.timedelta(days=1))
            add_extra_activities['basic_plan'] = np.where(add_extra_activities['same_date'] == True, add_extra_activities['basic_plan'], add_extra_activities['basic_plan_tomorrow'])
            add_extra_activities = add_extra_activities.drop(columns=["basic_plan_tomorrow"])


            # Remove the delays of the inserted events, such that they resemble a cancelled train
            add_extra_activities['basic_uitvoer'] = np.nan
            add_extra_activities['delay'] = np.nan
            add_extra_activities = add_extra_activities.drop(columns=["time"])
            # Combine the dataframe for the day with the extra activities
            df_r = pd.concat([df_res, add_extra_activities])
            # sort them
            df_r = df_r.sort_values(by=['date', 'basic_treinnr_treinserie', "basic_treinnr", 'basic_plan'])
            df_r = df_r.reset_index(drop=True)

            # add to the group again
            grouped_by_date[day_index] = df_r
    #create new datafame
    df_new = pd.concat(grouped_by_date)

    return df_new

def toSeconds(x : datetime.time):
    try:
        return x.total_seconds()
    except:
        return np.nan

def addbufferColumn(df):
    # sort the dataframe
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic_treinnr", "basic_plan"])
    # take the current plan and the plan of the previous event
    df['buffer'] = (df['basic_plan'] - df['basic_plan'].shift(1)).map(lambda x: x.total_seconds())
    # if those events are not at the same station, fill with 0
    df.loc[(df['basic_drp'] != df['basic_drp'].shift(1)) , 'buffer'] = 0
    # if those events are not between the same train, fill with 0
    df.loc[(df['basic_treinnr'] != df['basic_treinnr'].shift(1)) , 'buffer'] = 0
    # if those events are not at the same date, fill with 0
    df.loc[(df['date'] != df['date'].shift(1)), 'buffer'] = 0
    # for each event for which there was no 'upper' row to compare with, it is na, so also fill that with 0
    df['buffer'] = df.buffer.fillna(0)
    return df

def addTravelTimeColumn(df):
    # sort the dataframe
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic_treinnr", "basic_plan"])
    # take the current plan and the plan of the previous event and substract with buffer
    df['traveltime'] = (df['basic_plan'] - df['basic_plan'].shift(1)).map(lambda x: x.total_seconds())
    df['traveltime'] = df['traveltime']
    # if those events are not between the same train, fill with 0
    df.loc[(df['basic_treinnr'] != df['basic_treinnr'].shift(1)) , 'traveltime'] = np.nan
    # if those events are not at the same date, fill with 0
    df.loc[(df['date'] != df['date'].shift(1)), 'traveltime'] =np.nan
    return df

def removeCancelledTrain(df):
    # only keep the not-nan values
    df = df[df['vklvos_plan_actueel'].notna()]
    return df

def removeFacultativeTrains(df, df_sched):
    # get the trains that don't have a filled in basic uitvoer
    df_empty = copy.copy(df)
    df_empty = df_empty[df_empty['basic_uitvoer'].isna()]

    days_count = len(df.groupby('date'))
    # set treshold for amount of occurences
    min_occ = math.ceil(days_count * 0.35)

    g = df_empty.groupby(['basic_treinnr', 'basic_drp', 'basic_drp_act'])
    # only keep the events that occurs lower than the threshold, these are deleted later
    df_remove = g.filter(lambda x: len(x) > min_occ).reset_index(drop=True)
    df_remove = df_remove.drop_duplicates(subset=['basic_treinnr', 'basic_drp', 'basic_drp_act'], keep='first')

    return pd.concat([df_sched, df_remove, df_remove]).drop_duplicates(subset=['basic_treinnr', 'basic_drp', 'basic_drp_act'], keep=False).reset_index(drop=True)

def retrieveDataframe(export_name : str, workdays : bool, list_of_trainseries: List[str] = None, list_of_drp: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # split dataframes in column
    df = pd.read_csv(export_name, sep=";")
    print("Number of rows: ", len(df))
    df = df[
        ["nvgb_verkeersdatum", 'basic_treinnr_treinserie','basic_treinnr', 'basic_spoor_simpel', 'basic_drp', 'basic_drp_act', 'basic_plan', 'basic_uitvoer', 'vklvos_plan_actueel', "prl_rijw_laatst_rijweg_wisselroute_maxsnelheid", "basic_treinnr_rijkarakter", "vklbammat_soorttype_uitgevoerd", "vklvos_bijzonderheid_drglsnelheid", "donna_bd_kalerijtijd", "stipt_oorzaak_treinr"]]
    # set types of columns
    df['basic_treinnr_treinserie'] = df['basic_treinnr_treinserie'].astype('string')
    df['basic_drp'] = df['basic_drp'].astype('string')
    #df['basic_spoor_simpel'] = df['basic_spoor_simpel'].astype('string')
    df['basic_treinnr'] = df['basic_treinnr'].astype('string')
    df['basic_drp_act'] = df['basic_drp_act'].astype('string')
    df["basic_treinnr_rijkarakter"] = df["basic_treinnr_rijkarakter"].astype('string')
    df["vklbammat_soorttype_uitgevoerd"] = df[ "vklbammat_soorttype_uitgevoerd"].astype('string')
    df["vklbammat_soorttype_uitgevoerd"] = df["vklbammat_soorttype_uitgevoerd"].fillna("None")
    df['nvgb_verkeersdatum'] = pd.to_datetime(df['nvgb_verkeersdatum'], format='%Y-%m-%d').dt.date
    df['basic_plan'] = pd.to_datetime(df['basic_plan'], format='%Y-%m-%d %H:%M:%S')
    df['basic_uitvoer'] = pd.to_datetime(df['basic_uitvoer'], format='%Y-%m-%d %H:%M:%S')
    # use this column to determine if a train is cancelled
    df['vklvos_plan_actueel'] = pd.to_datetime(df['vklvos_plan_actueel'], format='%Y-%m-%d %H:%M:%S')
    # round the global plan to whole minutes
    df["global_plan"] = df['basic_plan'].dt.floor('Min')
    df["global_plan"] = df["global_plan"].dt.time
    df['delay'] = df['basic_uitvoer'] - df['basic_plan']
    df['delay'] = df['delay'].map(toSeconds)
    # if there is no basic uitvoer or basic plan, give a small random delay such that not everything is 0 or nan
    df['delay'] = df['delay'].apply(lambda v: random.randint(-5, 5) if np.isnan(v) else v)

    # rename column
    df['date'] = df['nvgb_verkeersdatum']
    df["speed"] = df['prl_rijw_laatst_rijweg_wisselroute_maxsnelheid']
    # The wissels (switches) are not implemented, but for a future implementation, this is already present.
    df = df.assign(wissels = "MP$283$R,MP$281B$R,MP$281A$R,MP$271A$L,MP$269$L,MP$263A$R,MP$247$R,MP$243B$R,MP$241A$L")
    df["stipt_oorzaak_treinr"] = df["stipt_oorzaak_treinr"].astype('string')
    df["stipt_oorzaak_treinr"] = df["stipt_oorzaak_treinr"].fillna("")

    # remove duplicated actions within de dataset
    df = df.drop_duplicates(subset=['date','basic_treinnr', 'basic_drp', 'basic_drp_act'],
                                                   keep='first')
    # only keep the desired trainseries or drps
    if list_of_trainseries != None:
        # only keep the desired train series
        df = keepTrainseries(df, list_of_trainseries)
    if list_of_drp != None:
        df = df[df['basic_drp'].isin(list_of_drp)]
    # add a buffer column
    df = addbufferColumn(df)
    # add a traveltime column
    df = addTravelTimeColumn(df)
    df['donna_bd_kalerijtijd'] = df['donna_bd_kalerijtijd'].fillna(0)
    df["slack"] = df['traveltime'] - df['donna_bd_kalerijtijd']
    # only keep the dates that are known
    df = df[~df['date'].isnull()]
    # only keep working days
    if workdays == None:
        pass
    elif workdays:
        df = keepWorkDays(df)
    elif not workdays:
        df = keepWeekendDays(df)

    # first remove the cancelled trains (since they do have a planning, delete them before the schedule is found, such that always cancelled trains are not included)
    df = removeCancelledTrain(df)

    # add a new column called "same date" which is an boolean containing information about if the date is the same as the date in the basic plan
    # if that is not the case, we know it is a night train.
    df["basic_plan_date"] = df.basic_plan.apply(lambda x: x.date())
    df["same_date"] = df["basic_plan_date"] == df["date"]
    df = df.drop(columns=["basic_plan_date"])
    # find the general schedule of the df
    sched = findSched(df)
    # according to this schedule, impute or remove entries
    df = makeDataUniform(df, sched)
    # after the schedule is created, the speed is ffilled over all the rows
    # the speed denotes the speed limit that is required for that specific event
    df["speed"] = df["speed"].fillna(
        method="ffill")
    df["speed"] = df["speed"].replace(
        "?", np.nan)
    # vklvos_bijzonderheid_drglsnelheid denotes the normal speed that is allowed.
    df["vklvos_bijzonderheid_drglsnelheid"] = df["vklvos_bijzonderheid_drglsnelheid"].fillna(method="ffill")
    # where is denoted that the speed is "baanvak", this means that the normal speed should be applied, which is saved in the vklvos_bijzonderheid_drglsnelheid column
    df['speed'] = np.where(
        df['speed'] == "baanvak", df['vklvos_bijzonderheid_drglsnelheid'],
        df['speed'])
    df["speed"] = df["speed"].astype('float')
    # sort the values
    df = df.sort_values(by=['date', 'basic_treinnr_treinserie', "basic_treinnr", "basic_plan"])
    df = df.reset_index(drop=True)

    return df[['date', 'basic_treinnr_treinserie','basic_treinnr', 'basic_spoor_simpel', 'basic_drp', 'basic_drp_act', "basic_plan" ,"global_plan", 'delay', "buffer", "traveltime", "wissels", "speed", "basic_treinnr_rijkarakter", "vklbammat_soorttype_uitgevoerd", "slack", "stipt_oorzaak_treinr"]], sched

def findSched(df):
    df_sched = copy.deepcopy(df)
    df_sched = df_sched.sort_values(by=['date', 'basic_treinnr_treinserie', "basic_treinnr", "basic_plan"])
    df_sched = df_sched.reset_index(drop=True)
    # get the amount of days
    days_count = len(df_sched.groupby('date'))
    print("Number of days: ", days_count)
    # set threshold for amount of occurences
    min_occ = math.ceil(days_count*0.35)
    g = df_sched.groupby(['basic_treinnr', 'basic_drp', 'basic_drp_act', "global_plan"])
    # only keep the events that occurs larger than the threshold
    df_sched = g.filter(lambda x: len(x) >= min_occ).reset_index(drop=True)

    # now we have all items that have a higher occurrence than the threshold from all days
    # since we want to have a schedule for one day, we keep all occurences once (now all different dates are included).
    df_sched = df_sched.drop_duplicates(subset=['basic_treinnr', 'basic_drp', 'basic_drp_act'], keep='first').reset_index(drop=True)
    print("events per day: ", len(df_sched))
    # add an old timestamp to it, to recognise the schedule entries
    timestamp = pd.to_datetime("01-01-2000", format='%d-%m-%Y')

    df_sched = df_sched.assign(date=timestamp)
    # update the basic_plan of the schedule
    df_sched = df_sched.assign(time=df_sched["basic_plan"].dt.time)
    df_sched.loc[:, "basic_plan"] = pd.to_datetime(df_sched.date.astype(str) + ' ' + df_sched.time.astype(str))

    #-------------------------------------------------------------------------------------------------- fix date jump in basic plan with the help of the "same date" column
    df_sched["basic_plan_tomorrow"] = df_sched["basic_plan"].apply(lambda x: x + dt.timedelta(days=1))
    df_sched['basic_plan'] = np.where(df_sched['same_date'] == True, df_sched['basic_plan'],df_sched['basic_plan_tomorrow'])
    df_sched = df_sched.drop(columns=["basic_plan_tomorrow"])
    # -------------------------------------------------------------------------------------------------- end fix date jump in basic plan done
    df_sched = df_sched.drop(columns=["time"])
    df_sched["delay"] = np.nan
    df_sched["speed"] = np.nan
    df_sched["nvgb_verkeersdatum"] = df_sched["date"]
    df_sched["stipt_oorzaak_treinr"] = ""
    df_sched= df_sched.sort_values(by=['basic_treinnr_treinserie', "basic_treinnr", "basic_plan"]).reset_index(drop=True)
    print(len(df_sched), "activities per day!")
    df_sched = removeFacultativeTrains(df, df_sched)
    # remove the freight trains and the trains that do not have a fixed schedule such as Losse loc
    df_sched = df_sched.loc[~df_sched.basic_treinnr_treinserie.str.startswith('GO', na=False)]
    df_sched = df_sched.loc[~df_sched.basic_treinnr_treinserie.str.startswith('LL', na=False)]
    df_sched = df_sched.reset_index(drop=True)
    print(len(df_sched), "activities per day, after removing the facultative, freight and ll trains!")
    return df_sched
