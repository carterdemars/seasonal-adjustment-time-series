import pandas as pd
from numpy import nan
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv


def preprocess_data(path):
    df = read_csv(path, sep=';', header=0, low_memory=False, infer_datetime_format=True,
                  parse_dates={'datetime': [0, 1]}, index_col=['datetime'])

    # mark all missing values
    df.replace('?', nan, inplace=True)

    # add a column for the remainder of sub metering
    values = df.values.astype('float32')
    df['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])

    # # aggregate daily
    df = df.astype('float32')
    df = df.groupby(df.index.date).agg('mean')

    # rename index
    df.index.rename('datetime', inplace=True)

    df.dropna(subset=['Global_active_power'], inplace=True)

    df.to_csv('household_power_consumption.csv')


