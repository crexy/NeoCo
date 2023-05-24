import json
import pyupbit
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
import time
import datetime
import cryptocode


#from strategy.Breakeout_strategy import Breakout_strategy
import os

# file_list = os.listdir("./strategy")
# strategy_file_list_py = [file for file in file_list if file.endswith(".py")]
# for file in strategy_file_list_py:
#     express = f"strategy.{}"
from strategy.Breakeout_strategy import *

class UpbitManage:

    balance = 1000000 # 잔액
    dict_coin_asset = {} # 코인자산
    dry_run = True # 임의 모드
    tradable_balance_ratio = 0.99 # 전체 자산 중 거래 가능 비율
    buy_mode = 0 # amount
    buy_amount = 100000 # 1회 매수 고정 금액
    buy_ratio = 0.3 # 1회 매수 금액 비율, 매수 금액 = 잔액 * stake_ratio
    tickers = [] # 티커(거래종목)
    strategy = None # 전략
    df_transaction_history = None # 걱래내역 장부

    # 거래 수수료
    trading_fee_rate = 0.0005

    # freqtrade timeframe 과 pyupbit timeframe 사전
    dict_timeframe_trans ={
        "1m": "minutes1",
        "5m": "minutes5",
        "1h": "minutes60",
        "4h": "minute240",
        "1d": "day"
    }

    def __init__(self, balance, tickers, strategy:str, stake_amount, stake_ratio=None, **kwargs):
        self.balace = balance # 초기 잔액
        self.tickers = tickers # 거래 종목들
        self.buy_mode = 2 if stake_amount == 0 else 1 # state_amount 값이 0 인경우 비율 매수 금액 모드(2) 아닌경우 고정 매수금액 모드(1)
        if self.buy_mode == 2: # 비율 매수 금액 모드 인 경우
            self.buy_ratio = stake_ratio

        self.strategy = eval(f"{strategy}()")

        if "trading_fee_rate" in kwargs:
            self.trading_fee_rate = kwargs.get("trading_fee_rate")
        if "tradable_balance_ratio" in kwargs:
            self.tradable_balance_ratio = kwargs.get("tradable_balance_ratio")

        # 거래내역 장부
        self.df_transaction_history = pd.DataFrame(
            columns=["date", "coin", "market", "act", "amount", "trans_amount", "fee", "profit_loss", "balance", "tot_asset", "ror"])
        self.df_transaction_history.set_index("date", inplace=True)


    # 액세스키, 비밀키 복호화
    def decrypt_apikeys(self):

        encryptkey = input("암호 키를 입력하세요: ")

        with open("./upbit_apikeys.key", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            encrypted_acckey = lines[0]
            encrypted_secretkey = lines[1]

        accesskey = cryptocode.decrypt(encrypted_acckey, encryptkey)
        secretkey = cryptocode.decrypt(encrypted_secretkey, encryptkey)

        return accesskey, secretkey


    def get_ohlcv(self, count):
        return pyupbit.get_ohlcv(count=200)

    def save_ohlcv(self, ticker="KRW-BTC", interval="day", count= 200):
        df_ohlcv = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count)
        df_ohlcv.to_csv("./data/ohlcv.csv")


    def load_ohlcv_json_file(self, ticker, timeframe):
        ticker = ticker.replace("/", "_")
        with open(f'./data/{ticker}-{timeframe}.json') as f:
            ohlcv_data = json.load(f)
        df = pd.DataFrame(ohlcv_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['buy'] = 0
        df['sell'] = 0
        df['hold'] = 0
        df['buy_price'] = 0
        df['sell_price'] = 0
        df['high_price'] = 0

        df['ror'] = 1 # Rate Of Return
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M'))
        return df

    def sell(self, sell_price, amount, idx, dataframe:DataFrame):
        dataframe.loc[idx, 'sell'] = amount
        dataframe.loc[idx, 'hold'] = dataframe.loc[idx, 'hold'] - amount
        dataframe.loc[idx, 'ror'] = dataframe.loc[idx, 'open'] / dataframe.loc[idx, 'buy_price']
        dataframe.loc[idx, 'buy_price'] = 0
        dataframe.loc[idx, 'sell_price'] = sell_price

    # 손절매
    # def stop_loss(self, cur_price, strategy, idx, dataframe: DataFrame):
    #
    #     # 보유 상태 확인
    #     if dataframe.loc[idx].hold == 0.0:
    #         return
    #
    #     # 손절매 플래그 확인
    #     if strategy.do_stop_loss == False:
    #         return
    #
    #     buy_price = dataframe.loc[idx].buy_price
    #
    #     if cur_price > buy_price * (1-strategy.stop_loss):
    #
    #     if strategy.trailing_stop_loss:
    #
    #
    #     if dataframe.loc[idx].hold == 1:


    def run_backtest(self):

        strategy = Breakout_strategy()

        tickers = ['BTC/USDT']

        strategy.tickers = tickers

        dict_ohlcv_org={}
        dict_ohlcv = {}

        for ticker in tickers:
            for timeframe in strategy.timeframes:
                #
                keystr = strategy.ohlcv_keystring(ticker, timeframe)
                dict_ohlcv_org[keystr] = self.load_ohlcv_json_file(ticker, timeframe)
                # Unnamed: 0 컬럼을 --> date 로 변경
                dict_ohlcv_org[keystr].rename(columns={dict_ohlcv_org[keystr].columns[0]: 'date'},
                                                   inplace=True)

        for k_val in np.arange(0.2, -0.25, -0.05):

            # load ohlcv data from file(json format)
            for ticker in tickers:
                for timeframe in strategy.timeframes:
                    #
                    keystr = strategy.ohlcv_keystring(ticker, timeframe)

                    strategy.dict_ohlcv[keystr] = dict_ohlcv_org[keystr].copy()

            #### populate_indicator and informative function running and merging ####

            # run populate_indicator function that return main timeframe's dataframe.

            # you have to merge main timeframe's dataframe and informative timeframe's dataframe

            # informative function is user custom function.
            # (*informative function format is "informative_[timeframe])
            # so for running informative function, first, confirm the informative function existing in strategy instance properties.
            # search informative function property in strategy class instance. you can get whole instance properties from 'dir([instance])' function.
            # if the function exist call the function by "eval()"
            # you can make informative function as string and run the string by "eval()" function.

            # it uses ticker as the key and store unified dataframe of the base timeframe and the informative timeframe.
            ticker_dict = {}

            prop_list = dir(strategy) # retrieve properties(variables and functions) of strategy instance

            for ticker in tickers:

                keystr = strategy.ohlcv_keystring(ticker, strategy.main_timeframe)
                main_df = strategy.dict_ohlcv[keystr]

                for timeframe in strategy.timeframes:
                    if timeframe != strategy.main_timeframe:
                        informative_func_str = f'informative_{timeframe}'
                        express = f'strategy.{informative_func_str}("timeframe", "{ticker}")'
                        # check if the informative function is actually declared.
                        if informative_func_str in prop_list:
                            # if it is declared, execute! as eval()
                            informative_df = eval(express)
                            # renaming informative dataframe columns as 'origin name + _timeframe'
                            # ex) if timeframe is '4h'
                            #     open -> open_4h
                            org_columns = informative_df.columns
                            new_columns = [x+"_"+timeframe for x in org_columns]
                            new_columns[0] = 'date'
                            informative_df.columns = new_columns
                            # merging main timeframe's dataframe and informative timeframe's dataframe
                            main_df = pd.merge(main_df, informative_df, left_on='date', right_on='date', how='left')
                            main_df = main_df.ffill()
                            meta_dict = {"ticker": ticker}
                            meta_dict['k_val'] = k_val # pass k_val
                            main_df = strategy.populate_indicator(main_df, meta_dict)
                main_df = main_df.dropna(axis=0)
                ticker_dict[ticker] = main_df

            keys = list(ticker_dict.keys())
            rows = ticker_dict[keys[0]].shape[0]

            pbar = tqdm(total=rows)

            for no in range(rows):
                pbar.update(1)
                for ticker in keys:
                    if no < 200:
                        continue

                    meta_dict={"ticker": ticker}

                    df = ticker_dict[ticker]
                    idx = df.index[no]

                    df.loc[idx, 'hold'] = df.loc[idx-1, 'hold']
                    df.loc[idx, 'buy_price'] = df.loc[idx-1, 'buy_price']

                    df_part = df.iloc[no-200:no]

                    # A value is trying to be set on a copy of a slice from a DataFrame.
                    # Try using .loc[row_indexer,col_indexer] = value instead
                    # df_part.hold = df_part.hold.ffill()

                    # stop loss checking
                    # self.stop_loss(df.loc[idx].open, strategy, idx, df)

                    do_buy = strategy.buy(df.loc[idx].open, df_part, meta_dict)

                    df.loc[idx, 'ror'] = 1

                    if do_buy:
                        df.loc[idx, 'buy'] = 1
                        df.loc[idx, 'hold'] = 1
                        df.loc[idx, 'buy_price'] = df.loc[idx].open

                    do_sell = strategy.sell(df.loc[idx].open, df_part, meta_dict)

                    if do_sell:
                        df.loc[idx, 'sell'] = 1
                        df.loc[idx, 'hold'] = 0
                        df.loc[idx, 'ror'] = df.loc[idx, 'open'] / df.loc[idx, 'buy_price']
                        df.loc[idx, 'buy_price'] = 0
                        df.loc[idx, 'sell_price'] = df.loc[idx].open

            pbar.close()

            for ticker in keys:

                df = ticker_dict[ticker]
                last_ror = df.ror.cumprod()

                print(f"\n========== {k_val:.2f} ==========")
                print(f"{ticker}: {last_ror.iloc[-1]:.2f}")


    # Convert Datetime index to column
    def convertDateIndexToCol(self, dataframe:DataFrame):
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={"index":"date"}, inplace=True)

    # merge main dataframe with other informative dataframe
    def merge_ohlcvDataFrames(self, ticker:str, meta_dict:dict=None):
        strategy_prop_list = dir(self.strategy)  # retrieve properties(variables and functions) of strategy instance

        keystr = self.strategy.ohlcv_keystring(ticker, self.strategy.main_timeframe)
        df_main = self.strategy.dict_ohlcv[keystr]

        for timeframe in self.strategy.timeframes:
            if timeframe != self.strategy.main_timeframe:

                keystr = self.strategy.ohlcv_keystring(ticker, timeframe)
                try:
                    # informative timeframe ohlcv data
                    self.strategy.dict_ohlcv[keystr] = pyupbit.get_ohlcv(ticker,
                                                                         self.dict_timeframe_trans[timeframe])
                    time.sleep(0.1)
                except:
                    print("Exception: infomative timeframe 'get_holcv()'!")
                    return False
                # Convert Datetime index to column
                self.convertDateIndexToCol(self.strategy.dict_ohlcv[keystr])
                # index를 date 컬럼으로 전환
                informative_func_str = f'informative_{timeframe}'
                express = f'self.strategy.{informative_func_str}("timeframe", "{ticker}")'
                # check if the informative function is actually declared.
                if informative_func_str in strategy_prop_list:
                    # if it is declared, execute! as eval()
                    informative_df = eval(express)
                    # renaming informative dataframe columns as 'origin name + _timeframe'
                    # ex) if timeframe is '4h'
                    #     open -> open_4h
                    org_columns = informative_df.columns
                    new_columns = [x + "_" + timeframe for x in org_columns]
                    new_columns[0] = 'date'
                    informative_df.columns = new_columns
                    # merging main timeframe's dataframe and informative timeframe's dataframe
                    df_merged = pd.merge(df_main, informative_df, left_on='date', right_on='date', how='left')




                    df_main = df_main.ffill()
                    df_main = self.strategy.populate_indicator(df_main, meta_dict)
        return True

    def buy_coin(self, dry_run:bool, coin:str, market:str, price:int) -> bool:

        buy_coin = 0.0 # 구매 수량
        if self.buy_mode == 0: # 고정 금액 매매 모드
            # 수수료
            fee = self.buy_amount * self.trading_fee_rate
            # check balance
            if self.balace < self.buy_amount + fee:
                return False
            buy_coin = self.buy_amount / price
            if dry_run:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
                new_date = pd.to_datetime(current_time)
                # create buy record.
                new_row={"coin": coin,
                         "market": market,
                         "act": "buy",
                         "amount": buy_coin,
                         "trans_amount": buy_coin*price,
                         "fee": fee,
                         "profit_loss": 0,
                         "balance": self.balace,
                         "tot_asset": 0,
                         "ror": 0}
                df_new_row = pd.DataFrame(new_row, index=[new_date])
                self.df_transaction_history = pd.concat([self.df_transaction_history, df_new_row])
            #else:
        else:
            buy_coin = (self.balace * self.buy_ratio) / price


        self.df_transaction_history

    # 거래 실행
    def run_trading(self, dry_run = True):

        # 업비트 주문 객체
        upbitOrder = pyupbit.Upbit(*self.decrypt_apikeys())

        while True:

            for ticker in self.tickers:

                # ticker에서 시장과 코인 분리
                market, coin = ticker.split("-")

                # ohlcv 데이터를 저장할 때 사용할 key 문자열을 종목과 시간봉을 조합하여 만든다.
                keystr = self.strategy.ohlcv_keystring(ticker, self.strategy.main_timeframe)

                try:
                    self.strategy.dict_ohlcv[keystr] = pyupbit.get_ohlcv(ticker,
                                                                         self.dict_timeframe_trans[self.strategy.main_timeframe],
                                                                         200)
                    time.sleep(0.1)
                except:
                    print("Exception: maintime frame 'get_holcv()'!")
                    continue

                # Convert Datetime index to column
                self.convertDateIndexToCol(self.strategy.dict_ohlcv[keystr])

                # merge main dataframe with other informative dataframe
                if self.merge_ohlcvDataFrames(ticker) == False:
                    continue

                # Get current price
                try:
                    cur_price = pyupbit.get_current_price(ticker)
                    time.sleep(0.1)
                except:
                    print("Exception: 'get_current_price()'")
                    print(f"Ticker: {ticker}")
                    continue

                # TODO: 매수 매도 시 로그 저장 방식/방법
                # 원칙1: 가장 간단한 방법 부터 검토 하여 구현 후 문제점 발견시 개선 및 refactoring 수행
                # 원칙2: 코드는 목적이 아니라 수단이다. 코드에 매몰되어 시간을 허비 하지 말자

                # get orderbook information
                orderbook = pyupbit.get_orderbook(ticker)

                df_main = self.strategy.dict_ohlcv[keystr]

                # check buy condition
                do_buy = self.strategy.buy(cur_price, df_main)

                if coin not in self.dict_coin_asset:
                    self.dict_coin_asset[coin] = 0.0

                # 보유 코인이 없을 경우에만 매수 가능
                if self.dict_coin_asset[coin] == 0.0:
                    if do_buy:
                        self.buy_coin(dry_run, coin, market, cur_price)


                # df.loc[idx, 'ror'] = 1
                #
                # if do_buy:
                #     df.loc[idx, 'buy'] = 1
                #     df.loc[idx, 'hold'] = 1
                #     df.loc[idx, 'buy_price'] = df.loc[idx].open

                # 보유코인이 있을 경우에만 매도 가능
                if self.dict_coin_asset[coin] > 0.0:
                    do_sell = self.strategy.sell(cur_price, df_main)

                # if do_sell:
                #     df.loc[idx, 'sell'] = 1
                #     df.loc[idx, 'hold'] = 0
                #     df.loc[idx, 'ror'] = df.loc[idx, 'open'] / df.loc[idx, 'buy_price']
                #     df.loc[idx, 'buy_price'] = 0
                #     df.loc[idx, 'sell_price'] = df.loc[idx].open


                time.sleep(0.1)
                print(orderbook)

            break



if __name__ == "__main__":
    # getAllTickers("KRW")
    # keys = decrypt_apikeys()
    # print("\naccess key:", keys[0])
    # print("secret key:", keys[1])

    # 1. ohlcv 데이터 다운로드
    # 2. 변동성 돌파 전략 작성
    # 3. backtest 진행
    # 4. hyperparameter 적용

    upbit = UpbitManage(1000000, ["KRW-BTC", "KRW-GMT"], "Breakout_strategy", 10000)
    upbit.run_trading(True)
    #upbit.run_backtest()

    #upbit.save_ohlcv(interval="minute5", count=57600)
    #upbit.load_ohlcv_json_file("BTC/USDT", "4h")
    #upbit.run_backtest()