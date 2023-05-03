from ta.trend import sma_indicator
import ccxt
import cryptocode
import sys
import datetime
import time
import pandas as pd

'''
1. 다운로드 ohlcv 
2. 
3.
'''

api_key = ''
api_secret = ''

def encrypt_apikeys(accesskey, secretkey):

    encryptkey = input("암호 키를 입력하세요: ")
    confirm_key = input("암호 키를 다시 입력하세요: ")

    if encryptkey != confirm_key:
        sys.exit("값이 다름니다.")

    print("입력한 key 값은: ",encryptkey,"입니다.")

    encrypted_acckey = cryptocode.encrypt(accesskey, encryptkey)
    encrypted_secretkey = cryptocode.encrypt(secretkey, encryptkey)

    with open("./upbit_apikeys.key", 'w', encoding='utf-8') as f:
        f.write(encrypted_acckey+"\n")
        f.write(encrypted_secretkey + "\n")

    print("save apikeys as ('upbit_apikeys.key')")

def decrypt_apikeys():

    encryptkey = input("암호 키를 입력하세요: ")

    with open("./upbit_apikeys.key", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        encrypted_acckey = lines[0]
        encrypted_secretkey = lines[1]

    accesskey = cryptocode.decrypt(encrypted_acckey, encryptkey)
    secretkey = cryptocode.decrypt(encrypted_secretkey, encryptkey)

    return (accesskey, secretkey)

exchange = ccxt.upbit(config={
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})
#
# print(exchange)

def getAllTickers(currency):
    tickers = exchange.fetch_tickers()
    symbols = tickers.keys()
    currency_symbols = [ x for x in symbols if x.endswith(currency)]
    #print(currency_symbols)
    return currency_symbols

def get_limit_ohlcv(symbol, timeframe):

    param ={
        'symbol': symbol,
        'timeframe': timeframe,
        'limit': 200
    }

    ohlcv = exchange.fetch_ohlcv(**param)
    df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    pd_ts = pd.to_datetime(df['datetime'], utc=True, unit='ms')  # unix timestamp to pandas Timeestamp
    pd_ts = pd_ts.dt.tz_convert("Asia/Seoul")  # convert timezone
    pd_ts = pd_ts.dt.tz_localize(None)
    df.set_index(pd_ts, inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def get_ohlcv(symbol, timeframe, since=None, to=None, tickCount=None):
    param ={
        'symbol': symbol,
        'timeframe': timeframe,
    }

    if tickCount is not None:
        param['limit'] = tickCount

    if since is not None:
        dt = datetime.datetime.strptime(since, "%Y-%m-%d")
        since_timestamp = datetime.datetime.timestamp(dt)
        param['since'] = int(since_timestamp)*1000

        timeframe_period = exchange.parse_timeframe(timeframe)

        if to is not None:
            dt = datetime.datetime.strptime(to, "%Y-%m-%d")
            to_timestamp = datetime.datetime.timestamp(dt)
            tickCount = (to_timestamp - since_timestamp) / timeframe_period
            param['limit'] = int(tickCount)
        else:
            if tickCount is None:
                tickCount = (time.time() - since_timestamp) / timeframe_period
                param['limit'] = int(tickCount)
    else:
        if tickCount is None:
            param['limit'] = 200

    ohlcv = exchange.fetch_premium_index_ohlcv(**param)
    df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    pd_ts = pd.to_datetime(df['datetime'], utc=True, unit='ms')  # unix timestamp to pandas Timeestamp
    pd_ts = pd_ts.dt.tz_convert("Asia/Seoul")  # convert timezone
    pd_ts = pd_ts.dt.tz_localize(None)
    df.set_index(pd_ts, inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


if __name__ == "__main__":
    # getAllTickers("KRW")
    # keys = decrypt_apikeys()
    # print("\naccess key:", keys[0])
    # print("secret key:", keys[1])

    # 1. ohlcv 데이터 다운로드
    # 2. 변동성 돌파 전략 작성
    # 3. backtest 진행
    # 4. hyperparameter 적용

    ohlcv = get_ohlcv("BTC/KRW", "5m", "2022-01-01", "2023-04-01")
    ohlcv.to_csv("./data/btc_krw_5m.csv")
    print(ohlcv)


