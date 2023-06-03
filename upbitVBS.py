import json
import pyupbit
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
import time
import datetime
import cryptocode
from threading import Thread
import math
# VBS(Volatility Breakout Strategy) Larry R. Williams

# 매수 시 고정값 및 비율 값
FIX_BUY = 1
RATIO_BUY = 2

class upbitVBS:

    balance = 1000000  # 잔액
    # ticker {"amount":0.0, "buy_price":0.0, "high_price":0.0, "cur_price":0.0} 구조
    dict_coin_asset = {}  # 코인자산

    dry_run = True  # 임의 모드
    tradable_balance_ratio = 0.99  # 전체 자산 중 거래 가능 비율

    buy_mode = 0  # amount
    buy_amount = 100000  # 1회 매수 고정 금액
    buy_ratio = 0.3  # 1회 매수 금액 비율, 매수 금액 = 잔액 * stake_ratio
    tickers = []  # 티커(거래종목)
    df_transaction_history = None  # 걱래내역 장부

    # 거래 수수료
    trading_fee_rate = 0.0005

    # 손절매
    stop_loss_flg = True
    stop_loss_val = 0.03

    # 추적 손절매 플래그
    trailing_stop_flg = False

    # 최대 실행 트레이딩 갯수
    max_open_trade = 3

    # 트레이딩 실행 플래그
    trading_run_flg = True

    # 시간주기
    #["day", "days"]:
    #["minute1", "minutes1"]
    #["minute3", "minutes3"]
    #["minute5", "minutes5"]
    #["minute10", "minutes10"]:
    #["minute15", "minutes15"]:
    #["minute30", "minutes30"]:
    #["minute60", "minutes60"]:
    #["minute240", "minutes240"]:
    #["week",  "weeks"]:
    #["month", "months"]:
    #interval = "day"

    def __init__(self, balance, tickers, interval, stake_amount, stake_ratio=None, stop_loss_flg=True, **kwargs):
        self.balance = balance  # 초기 잔액
        self.tickers = tickers  # 거래 종목들

        self.buy_amount = stake_amount
        self.buy_mode = RATIO_BUY if stake_amount == 0 else FIX_BUY  # state_amount 값이 0 인경우 비율 매수 금액 모드(2) 아닌경우 고정 매수금액 모드(1)
        if self.buy_mode == RATIO_BUY:  # 비율 매수 금액 모드 인 경우
            self.buy_ratio = stake_ratio

        self.interval = interval # 타임프레임

        # 손절매 플래그
        self.stop_loss_flg = stop_loss_flg

        if "trading_fee_rate" in kwargs:
            self.trading_fee_rate = kwargs.get("trading_fee_rate")
        if "tradable_balance_ratio" in kwargs:
            self.tradable_balance_ratio = kwargs.get("tradable_balance_ratio")
        if "stop_loss" in kwargs:
            self.stop_loss_val = kwargs.get("stop_loss")
        if "trailing_stop" in kwargs:
            self.trailing_stop_flg = kwargs.get("trailing_stop")
        if "max_open_trade" in kwargs:
            self.max_open_trade = kwargs.get("max_open_trade")

        # 거래내역 장부
        self.df_transaction_history = pd.DataFrame(
            columns=["date", "ticker", "act", "volume", "price", "fee", "order_uuid", "balance", "stop_loss", "loss_gain"])
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

    # 주문 로그 작성
    def write_order_log(self, ticker, act, volume, price, fee, order_uuid, order_date, stop_loss=0, loss_gain=1):
        # create buy record.
        new_row = {"ticker": ticker,
                   "act": act,
                   "volume": volume,
                   "price": price,
                   "fee": fee,
                   "order_uuid": order_uuid,
                   "balance": self.balance,
                   "stop_loss": 1 if stop_loss else 0,
                   "loss_gain": loss_gain}
        df_new_row = pd.DataFrame(new_row, index=[order_date])
        self.df_transaction_history = pd.concat([self.df_transaction_history, df_new_row])

    # SSS================= 보유 자산 정보 설정 함수 =================SSS
    def init_coin_asset_info(self, ticker):
        self.dict_coin_asset[ticker] = {"volume":0.0, "cur_price":0.0, "buy_price":0.0, "high_price":0.0}

    def get_coin_volume(self, ticker) -> float:
        if ticker in self.dict_coin_asset == False: # 코인 내역이 없다면
            self.init_coin_asset_info(ticker)
        return self.dict_coin_asset[ticker]["volume"]

    def get_coin_value(self, ticker) -> float:
        return self.dict_coin_asset[ticker]["cur_price"] * self.dict_coin_asset[ticker]["volume"]

    def get_total_coin_value(self):
        total_coin_value = 0.0
        for ticker in self.tickers:
            total_coin_value += self.get_coin_value(ticker)
        return total_coin_value
    def has_coin(self, ticker)->bool:
        return True if self.get_coin_volume(ticker) > 0 else False

    def set_coin_cur_price(self, ticker, cur_price):
        self.dict_coin_asset[ticker]["cur_price"] = cur_price
        if cur_price > self.dict_coin_asset[ticker]["high_price"]:
            self.dict_coin_asset[ticker]["high_price"] = cur_price

    def set_buy_coin(self, ticker, price, volume):
        if ticker in self.dict_coin_asset == False: # 코인 내역이 없다면
            self.init_coin_asset_info(ticker)
        self.dict_coin_asset[ticker]["volume"] = volume
        self.dict_coin_asset[ticker]["cur_price"] = price
        self.dict_coin_asset[ticker]["buy_price"] = price
        self.dict_coin_asset[ticker]["high_price"] = price

    def get_coin_buy_price(self, ticker):
        if ticker in self.dict_coin_asset == False: # 코인 내역이 없다면
            self.init_coin_asset_info(ticker)
        return self.dict_coin_asset[ticker]["buy_price"]

    def get_coin_high_price(self, ticker):
        if ticker in self.dict_coin_asset == False: # 코인 내역이 없다면
            self.init_coin_asset_info(ticker)
        return self.dict_coin_asset[ticker]["high_price"]
    # EEE================= 보유 자산 정보 설정 함수 =================EEE

    # 손절매 수행
    def stop_loss(self, dryrun, ticker, cur_price, current_time=None):
        if self.stop_loss_flg == False: return # 손절매 플래그 확인
        if self.has_coin(ticker) == False: return# 코인을 소지하지 않았다면

        if self.trailing_stop_flg: # 추적 손절매
            std_price = self.get_coin_high_price(ticker)
        else: #손절매
            std_price = self.get_coin_buy_price(ticker)
        # 기준가(구매가/최고가) 대비 현재가가 손절매 비율 보다 높다면 자산 손절매
        if ((cur_price / std_price)-1) > self.stop_loss_val:
            self.sell_coin(dryrun, ticker, cur_price, True, current_time)

    # 코인 매수
    def buy_coin(self, dryrun:bool, ticker, price:float, current_time=None) -> bool:

        buy_volume = 0.0 # 구매 수량
        order_uuid = "0" # 주문번호
        if self.buy_mode == FIX_BUY: # 고정 금액 매매 모드
            # 수수료
            fee = self.buy_amount * self.trading_fee_rate
            # check balance
            if self.balance < (self.buy_amount + fee)*1.1:
                return False
            buy_volume = self.buy_amount / price
        else:
            buy_volume = (self.balance * self.buy_ratio) / price
            fee = buy_volume * price * self.trading_fee_rate

            # 최소 매수 금액
            if buy_volume * price < 5000:
                return False

            # check balance
            if self.balance < (buy_volume * price + fee)*1.1:  # 잔고 확인
                return False

        if dryrun == False: # 실거래
            # 시장가 주문
            rslt = self.upbit.buy_market_order(ticker, buy_volume * price, True)
            price = rslt["price"]
            if rslt["executed_volume"] != buy_volume:
                executed_volume = rslt["executed_volume"]
                print(f"Info: market order remaining) order volume: {buy_volume}, excuted_volume: {executed_volume}")
            buy_volume = rslt["executed_volume"]
            fee = rslt["paid_fee"]
            order_uuid = rslt["uuid"]

        # 주문 시간
        if current_time is None:
            current_time = datetime.datetime.now()

        # 매수 코인 자산 정보 갱신
        self.set_buy_coin(ticker, price, buy_volume)
        # 잔액 갱신
        self.balance -= (buy_volume * price + fee)

        # create buy record.
        self.write_order_log(ticker, "buy", buy_volume, price, fee, order_uuid, current_time)

        return True

    # 코인 매도
    def sell_coin(self, dryrun:bool, ticker:str, price:float, bstop_loss:bool=False, current_time=None) -> bool:

        # 매도 수량
        sell_volume = self.get_coin_volume(ticker)

        #current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #new_date = pd.to_datetime(current_time)
        order_uuid = "0" # 주문번호
        fee = price * sell_volume * self.trading_fee_rate # 수수료

        if dryrun == False: # 실거래
            # 시장가 주문
            rslt = self.upbit.sell_market_order(ticker, sell_volume, True)
            price = rslt["price"]
            sell_volume = rslt["executed_volume"]
            fee = rslt["paid_fee"]
            order_uuid = rslt["uuid"]

        # 주문 시간
        if current_time is None:
            current_time = datetime.datetime.now()

        # 매수 코인 자산 초기화
        self.init_coin_asset_info(ticker)
        # 잔액 갱신
        self.balance += (sell_volume * price - fee)

        # 매수 매도 손익률 구하기
        mask = (self.df_transaction_history["ticker"] == ticker) & (self.df_transaction_history["act"] == "buy")
        df_history_buy = self.df_transaction_history.loc[mask,:]
        buy_price = df_history_buy.iloc[-1].price
        buy_volume = df_history_buy.iloc[-1].volume
        buy_fee = df_history_buy.iloc[-1].fee
        loss_gain = (price * sell_volume - fee) / (buy_price * buy_volume + buy_fee)

        # create sell record.
        self.write_order_log(ticker, "sell", sell_volume, price, fee, order_uuid, current_time, bstop_loss, loss_gain)

    #시간간격을 timedelta 객체로
    @staticmethod
    def interval_to_timedelta(interval):
        if interval == "day":
            timedelta = datetime.timedelta(days=1)
        elif interval == "minute1":
            timedelta = datetime.timedelta(minutes=1)
        elif interval == "minutes3":
            timedelta = datetime.timedelta(minutes=3)
        elif interval == "minutes5":
            timedelta = datetime.timedelta(minutes=5)
        elif interval == "minutes10":
            timedelta = datetime.timedelta(minutes=10)
        elif interval == "minutes15":
            timedelta = datetime.timedelta(minutes=15)
        elif interval == "minutes30":
            timedelta = datetime.timedelta(minutes=30)
        elif interval == "minutes60":
            timedelta = datetime.timedelta(hours=1)
        elif interval == "minutes240":
            timedelta = datetime.timedelta(hours=4)
        elif interval == "weeks":
            timedelta = datetime.timedelta(weeks=1)
        elif interval == "months":
            timedelta = datetime.timedelta(weeks=4)
        return timedelta

    def trading(self, dryrun, k, skip_minuts=1):
        # 업비트 주문 객체
        self.upbit = pyupbit.Upbit(*self.decrypt_apikeys())

        if dryrun == False:
            # 잔액 및 보유 코인 정보 요청
            pass

        while self.trading_run_flg:
            dict_price={} # 현재가

            for ticker in self.tickers:
                try:
                    df_ohlcv = pyupbit.get_ohlcv(ticker, self.interval, 2)
                    cur_price = df_ohlcv.loc[df_ohlcv.index[-1], "close"] # 현재가
                    dict_price[ticker] = cur_price
                    #time.sleep(0.1)
                except:
                    print("Exception: maintime frame 'get_holcv()'!")
                    continue

                # 손절매 코드
                self.stop_loss(dryrun, ticker, cur_price)

                # 현재 시간
                current_time = datetime.datetime.now()
                # 시작 시간
                start_time = df_ohlcv.index[1]
                # 종료 시간

                end_time = start_time + self.interval_to_timedelta(self.interval)

                # 기준 시가 09:00 일 경우, 금일 09:00 ~ 명일 09:00 - 20초 까지의 기간은 매수 가능 시간
                if start_time < current_time < end_time - datetime.timedelta(minutes=skip_minuts):
                    # 기준 가격 계산
                    target_price = df_ohlcv.iloc[0]['close'] + (df_ohlcv.iloc[0]['high'] - df_ohlcv.iloc[0]['low']) * k
                    # 현재가가 기준 가격보다 상승한다면
                    if target_price < cur_price: # 매수 신호
                        # 보유 코인이 없을 경우에만 매수 가능
                        if self.has_coin(ticker) == False:
                            self.buy_coin(dryrun, ticker, cur_price)
                else: # 그 이외의 시간 (20초), 시간 기준이 전환 되는 시간
                    # 보유 코인이 있을 경우만 매도 가능
                    if self.has_coin(ticker):
                        self.sell_coin(dryrun, ticker, cur_price)

            # 전체 자산 계산
            time.sleep(1)

    @staticmethod
    def change_interval_word(interval):
        if interval == "day":
            return "1d"
        elif interval == "minute1":
            return "1m"
        elif interval == "minutes3":
            return "3m"
        elif interval == "minutes5":
            return "5m"
        elif interval == "minutes10":
            return "10m"
        elif interval == "minutes15":
            return "15m"
        elif interval == "minutes30":
            return "30m"
        elif interval == "minutes60":
            return "1h"
        elif interval == "minutes240":
            return "4h"

        return None

    # 백테스트 리포트
    def backtest_report(self):
        print("=========================================================================")
        print("=                             백테스트 결과                               =")
        print("=========================================================================")
        idx = self.df_transaction_history.index[0]
        init_asset_value = self.df_transaction_history.loc[idx, "balance"] + \
                       self.df_transaction_history.loc[idx, "price"] * \
                       self.df_transaction_history.loc[idx, "volume"] + \
                       self.df_transaction_history.loc[idx, "fee"]
        laset_asset_value = self.balance + self.get_total_coin_value()
        print(f"초기 잔액: {init_asset_value:.2f}")
        print(f"최종 잔액: {self.balance:.2f}")
        print(f"수 익 률: {laset_asset_value/init_asset_value*100:.2f}%")

        print("\n")
        for ticker in self.tickers:
            # ticker 별 수익률
            mask = (self.df_transaction_history["ticker"] == ticker) & (self.df_transaction_history["act"] == "sell")
            df = self.df_transaction_history.loc[mask, :]
            returns = df.loss_gain.cumprod().iloc[-1]
            print(f"{ticker} 수익률: {returns*100:.2f}%")
        print("=========================================================================")

    # ohlcv 파일 데이터 로드
    def load_ohlcv_json_file(self, path, ticker, interval):
        market, coin = ticker.split("-")
        timeframe = self.change_interval_word(interval)
        if timeframe is None:
            return None

        if path[-1] == "/" or path[-1] == "\\":
            path = path[:-1]
        with open(f'{path}/{coin}_{market}-{timeframe}.json') as f:
            ohlcv_data = json.load(f)
        df = pd.DataFrame(ohlcv_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S'))
        return df



    # 백테스팅
    def backtesting(self, k=1.2, filepath="./data", skip_minuts=1):

        dict_ohlcv = {} # 각 ticker 별 ohlcv 데이터
        pbar = tqdm(desc="Loading OHLCV", total=len(self.tickers))

        # 데이터 개수
        data_cnt = 1000000000000

        for ticker in self.tickers:
            df = self.load_ohlcv_json_file(filepath, ticker, "minute1")
            df_base = df[["date", "open"]].copy()
            df_base.rename(columns={"open":"price"}, inplace=True)
            df_ohlcv = self.load_ohlcv_json_file(filepath, ticker, self.interval)
            # timestamp
            s_date = pd.to_datetime(df_ohlcv.date)
            df_ohlcv['timestamp'] = s_date.astype(np.int64)
            # target price
            df_ohlcv['trg_price'] = (df_ohlcv['high'].shift(1) - df_ohlcv['low'].shift(1)) * k + df_ohlcv['open']

            df_merged = pd.merge(df_base, df_ohlcv, left_on='date', right_on='date', how='left')

            df_merged = df_merged.ffill()
            # flag period changed row
            s = df_merged['timestamp'] - df_merged['timestamp'].shift(1)
            df_merged['period_changed'] = np.where(s > 0, 1, 0)
            df_merged.dropna(axis=0, inplace=True)

            dict_ohlcv[ticker] = df_merged

            # Asset info initiation
            self.init_coin_asset_info(ticker)

            # 각 티커의 ohlcv 데이터의 길이가 다를 경우를 대비해 loop의 기준을 가장 적은 데이터를 기준으로 한다.
            if df_merged.shape[0] < data_cnt:
                data_cnt = df_merged.shape[0]

            pbar.update(1)

        pbar.close()

        # 프로그레스바 객체
        pbar = tqdm(desc=f"Backtesting k-val:{k}", total=data_cnt)

        for no in range(data_cnt):
            # 이전 티커의 시간 저장 플래그
            prev_ticker_time = ""

            for ticker in self.tickers:
                idx = dict_ohlcv[ticker].index[no]

                # 각 티커의 ohlcv 개별 행 데이터 날짜 일치여부 확인
                if prev_ticker_time != "":
                    if prev_ticker_time != dict_ohlcv[ticker].loc[idx, "date"]:
                        print("TIME SYNC ERROR!!")
                        return

                prev_ticker_time = dict_ohlcv[ticker].loc[idx, "date"]

                # current price
                cur_price = dict_ohlcv[ticker].loc[idx, "price"]

                # period change flag
                period_changed = dict_ohlcv[ticker].loc[idx, "period_changed"]

                # 현재 시간
                current_time = datetime.datetime.strptime(dict_ohlcv[ticker].loc[idx, "date"], '%Y-%m-%d %H:%M:%S')

                # 시작 시간
                if period_changed == 1 or no == 0:
                    start_time = datetime.datetime.strptime(dict_ohlcv[ticker].loc[idx, "date"], '%Y-%m-%d %H:%M:%S')

                # 종료 시간
                end_time = start_time + self.interval_to_timedelta(self.interval)

                # 손절매 코드
                self.stop_loss(True, ticker, cur_price, current_time)

                # 기준 시가 09:00 일 경우, 금일 09:00 ~ 명일 09:00 - 20초 까지의 기간은 매수 가능 시간
                if start_time <= current_time < (end_time - datetime.timedelta(minutes=skip_minuts)):
                    # 기준 가격 계산
                    target_price = dict_ohlcv[ticker].loc[idx, "trg_price"]
                    # 현재가가 기준 가격보다 상승한다면
                    if target_price < cur_price: # 매수 신호
                        # 보유 코인이 없을 경우에만 매수 가능
                        if self.has_coin(ticker) == False:
                            self.buy_coin(True, ticker, cur_price, current_time)
                else: # 그 이외의 시간 (20초), 시간 기준이 전환 되는 시간
                    # 보유 코인이 있을 경우만 매도 가능
                    #print(ticker+":"+str(self.has_asset(ticker)))
                    if self.has_coin(ticker):
                        self.sell_coin(True, ticker, cur_price, False, current_time)

            pbar.update(1)
        pbar.close()

        # 백테스트 리포트 출력
        self.backtest_report()
        # 백테스트 데이터 저장


if __name__ == "__main__":
    # getAllTickers("KRW")
    # keys = decrypt_apikeys()
    # print("\naccess key:", keys[0])
    # print("secret key:", keys[1])

    # 1. ohlcv 데이터 다운로드
    # 2. 변동성 돌파 전략 작성
    # 3. backtest 진행
    # 4. hyperparameter 적용

    upBitVBS = upbitVBS(1000, ["USDT-BTC", "USDT-ETH", "USDT-GALA"], "day", 300)
    #upBitVBS = upbitVBS(1000, ["USDT-GALA"], "day", 300)
    upBitVBS.backtesting(1.6, "./data/binance/debug", 60)
    #upBitlarryW.trading(True, 1)

    # dict_price = pyupbit.get_current_price(["KRW-BTC", "KRW-GMT"])
    # print("KRW-BTC:"+str(dict_price["KRW-BTC"]))
    #
    # df = pyupbit.get_ohlcv("KRW-BTC", "day", 2)
    # idx = df.index[-1]
    # print("ohlcv close: "+str(df.loc[idx, "close"]))
