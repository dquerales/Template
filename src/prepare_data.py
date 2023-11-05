import pandas as pd
from logger import logger, log_data_status

def add_date_features(dataframe, date):
    df = dataframe.copy()
    logger('Adding data features')
    log_data_status(df)
    df[date] = pd.to_datetime(df[date])
    df['hour'] = df[date].dt.hour
    df['dayofweek'] = df[date].dt.dayofweek
    df['quarter'] = df[date].dt.quarter
    df['month'] = df[date].dt.month
    df['year'] = df[date].dt.year
    df['dayofyear'] = df[date].dt.dayofyear
    df['dayofmonth'] = df[date].dt.day
    df = df.set_index(date).sort_index()
    logger('features added.')
    log_data_status(df)
    return(df)
