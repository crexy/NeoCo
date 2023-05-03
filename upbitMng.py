import json
import pyupbit
import pandas as pd


from Breakeout_strategy import Breakout_strategy

class UpbitManage:

    #def UpbitManage(self):

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
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M'))
        return df

    def run_backtest(self):

        # ohlcv data dictionary


        strategy = Breakout_strategy()

        tickers = ['BTC/USDT']

        strategy.tickers = tickers

        # load ohlcv data from file(json format)
        for ticker in tickers:
            for timeframe in strategy.timeframes:
                #
                keystr = Breakout_strategy.ohlcv_keystring(ticker, timeframe)
                strategy.dict_ohlcv[keystr] = self.load_ohlcv_json_file(ticker, timeframe)
                # Unnamed: 0 컬럼을 --> date 로 변경
                strategy.dict_ohlcv[keystr].rename(columns={strategy.dict_ohlcv[keystr].columns[0]: 'date'}, inplace=True)

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

            keystr = Breakout_strategy.ohlcv_keystring(ticker, strategy.main_timeframe)
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
                        main_df = strategy.populate_indicator(main_df, meta_dict)
            main_df = main_df.dropna(axis=0)
            ticker_dict[ticker] = main_df

        keys = list(ticker_dict.keys())
        rows = ticker_dict[keys[0]].shape[0]

        for no in range(rows):
            for ticker in keys:
                if no < 200:
                    continue

                meta_dict={"ticker": ticker}

                df = ticker_dict[ticker]
                idx = df.index[no]

                do_buy = strategy.buy(df.iloc[no], df.iloc[no-200:no], meta_dict)

                if do_buy:
                    df.loc[idx, 'buy'] = 1

                do_sell = strategy.sell(df.iloc[no], df.iloc[no - 200:no], meta_dict)

                if do_sell:
                    df.loc[idx, 'sell'] = 1























if __name__ == "__main__":
    # getAllTickers("KRW")
    # keys = decrypt_apikeys()
    # print("\naccess key:", keys[0])
    # print("secret key:", keys[1])

    # 1. ohlcv 데이터 다운로드
    # 2. 변동성 돌파 전략 작성
    # 3. backtest 진행
    # 4. hyperparameter 적용

    upbit = UpbitManage()
    #upbit.run_backtest()

    #upbit.save_ohlcv(interval="minute5", count=57600)
    #upbit.load_ohlcv_json_file("BTC/USDT", "4h")
    upbit.run_backtest()