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

    timeframes = ['5m', '4h']
    main_timeframe = '5m'
    # ohlcv data dictionary, it is defined from outside.
    dict_ohlcv = {}
    # tickers list, it is defined from outside.
    tickers=[]

    # return ticker_timeframe key string ex) BTC/KRW, 5m --> BTC_KRW_5m
    @staticmethod
    def ohlcv_keystring(ticker, timeframe):
        keystr = ticker.replace("/", "_") + "-" + timeframe
        return keystr

    # 데코레이터


    @informative('4h')
    def informative_4h(self, dataframe: DataFrame, ticker: str):
        s_date = pd.to_datetime(dataframe.date)
        dataframe['timestamp'] = s_date.astype(np.int64)

        return dataframe


    def populate_indicator(self, dataframe: DataFrame, metadata: dict=None) -> DataFrame:
        s = dataframe['timestamp_4h'] - dataframe['timestamp_4h'].shift(1)
        dataframe['time_changed'] = np.where(s > 0, 1, 0)

        larry_K = 0.6
        # 변동성 돌파 전략 기준 값 설정
        dataframe['trg_price'] = (dataframe['high_4h'].shift(1) - dataframe['low_4h'].shift(1)) * larry_K + dataframe['open_4h']
        dataframe['trg_price'] = np.where(dataframe['time_changed'] == 1, dataframe['trg_price'], np.nan)
        dataframe['trg_price'] = dataframe['trg_price'].ffill()

        return dataframe

    # 매수 함수
    def buy(self, cur_ohlcv: Series, dataframe: DataFrame, metadata: dict):

        if cur_ohlcv.open > cur_ohlcv.trg_price and cur_ohlcv.volume > 0:
            # TODO: 기준 타임프레임 중 중복 매수 방지 코드 필요
            return True

        return False


    # 매도 함수
    def sell(self, cur_ohlcv: Series, dataframe: DataFrame, metadata: dict):

        if cur_ohlcv.time_changed == 1:
            return True

        return False