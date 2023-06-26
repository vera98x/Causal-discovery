from datetime import datetime, time
from enum import Enum
import numpy as np
import pandas as pd
from Utils import nan_equal

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


filename = "Data/Ontkoppelpuntenlijst.csv"
df = pd.read_csv(filename, sep=";")
df = df[["Dienstregelpunt", "Logistieke functionaliteit voor VGMI database"]]
df = df.rename(columns={"Dienstregelpunt": "drp", "Logistieke functionaliteit voor VGMI database": "type"})
station_dict = df.set_index('drp').T.to_dict("list")


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



filename = "Data/Dienstregelpuntverbindingen_afstanden.csv"
df = pd.read_csv(filename, sep=";")
df = df[["Van", "Naar", "Aantal hectometers"]]
df["Aantal hectometers"] = df["Aantal hectometers"].astype('int')
distance_dict = df.groupby('Van')[['Naar','Aantal hectometers']].apply(lambda x: x.set_index('Naar').to_dict(orient='index')).to_dict()

class NN_input_row:
    def __init__(self, tro, prev_event_tro, row_info, time_interval):
        global station_dict
        global minimal_dwell_time
        global distance_dict
        self.label = tro.getSmallerID()
        self.drp = tro.getStation()
        self.date = tro.date #row_info[0]
        self.tro_delay = row_info[0]
        self.dep_delay = row_info[1:] # consists of 6 items prev_event, prev_prev_event, prev_platform and 3 other delays
        self.timeinterval = time_interval
        self.type_train = tro.getTraintype()
        self.train_mat = tro.getTrainmat()
        self.type_station = station_dict[tro.getStation()]
        self.day_of_the_week = self.getDay(self.date)
        self.peakhour = self.getPeak(tro.getPlannedTime_time())
        self.buffer = tro.getBuffer()
        self.traveltime = tro.getTraveltime()
        self.trainserie = tro.getTrainSerie()
        self.activity = tro.getActivity()
        self.minimal_dwell_time = minimal_dwell_time[tro.getStation()].get(self.type_train, np.nan)
        self.speed = tro.getSpeed()
        self.slack = tro.getSlack()
        self.cause = tro.getCause()
        if prev_event_tro == None:
            self.traveldistance = np.nan
        elif tro.getStation() == prev_event_tro.getStation():
            self.traveldistance = 0
        else:
            # on purpose no default value, since it should give an error when a distance is not known
            self.traveldistance = distance_dict[tro.getStation()][prev_event_tro.getStation()]["Aantal hectometers"]

    def __eq__(self, other):
        a = np.array(self.dep_delay)
        b = np.array(other.dep_delay)
        return (self.label == other.label and self.drp == other.drp and self.date == other.date and ((self.tro_delay == other.tro_delay) or (np.isnan(self.tro_delay) and np.isnan(other.tro_delay))) and nan_equal(a, b) and self.timeinterval == other.timeinterval)

    def __repr__(self):
        return (str(self.label) + "_+" + str(self.drp) + "_+" + str(self.date) + "_+" + str(self.tro_delay) + "_+" + str(self.dep_delay) + "_+" + str(self.timeinterval))

    def getDay(self, tro_date : datetime.date) -> Weekday:
        day = tro_date.weekday()
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