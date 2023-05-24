import pandas as pd
from pandas import DataFrame, Series
import numpy as np


def informative(timeframe):
    def decorator(func):
        def wrapper(cls, dataframe, ticker):
            keystr = cls.ohlcv_keystring(ticker, timeframe)
            return func(cls, cls.dict_ohlcv[keystr], ticker)
        return wrapper
    return decorator

class Breakout_strategy:

    timeframes = ['1m', '1d']
    main_timeframe = '1m'

    # essential val be used for informative function
    # essential val
    dict_ohlcv={}

    # 손절매 플래그
    do_stop_loss = True
    # 손절매
    stop_loss = 0.1

    # 추적 손절매 플래그
    do_trailing_stop_loss = True
    # 추적 손절매
    trailing_stop_loss = 0.05

    # return ticker_timeframe key string ex) BTC/KRW, 5m --> BTC_KRW_5m
    # essential val be used for informative function
    @staticmethod
    def ohlcv_keystring(ticker, timeframe):
        keystr = ticker.replace("-", "_") + "-" + timeframe
        return keystr

    # 데코레이터

    @informative('1d')
    def informative_1d(self, dataframe: DataFrame, ticker: str):
        s_date = pd.to_datetime(dataframe.date)
        dataframe['timestamp'] = s_date.astype(np.int64)

        return dataframe

    def populate_indicator(self, dataframe: DataFrame, metadata: dict=None) -> DataFrame:
        s = dataframe['timestamp_1d'] - dataframe['timestamp_1d'].shift(1)
        dataframe['time_changed'] = np.where(s > 0, 1, 0)

        larry_k = 0.1
        if metadata is not None:
            larry_k = metadata['k_val']
        # 변동성 돌파 전략 기준 값 설정
        dataframe['trg_price'] = (dataframe['high_1d'].shift(1) - dataframe['low_1d'].shift(1)) * larry_k + dataframe['open_1d']
        dataframe['trg_price'] = np.where(dataframe['time_changed'] == 1, dataframe['trg_price'], np.nan)
        dataframe['trg_price'] = dataframe['trg_price'].ffill()

        return dataframe

    # 매수 함수
    def buy(self, cur_price: int, dataframe: DataFrame, metadata: dict=None):
        if cur_price > dataframe.iloc[-1].trg_price and \
                dataframe.iloc[-1].volume > 0 and \
                dataframe.iloc[-1].hold == 0:
            return True
        return False


    # 매도 함수
    def sell(self, cur_price: int, dataframe: DataFrame, metadata: dict):
        if dataframe.iloc[-1].time_changed == 1 and \
                dataframe.iloc[-1].hold == 1:
            return True
        return False