import numpy as np
def get_coin_info(coin_name, minute='day'):
    lookback = 5
    count = 200
    time_row = 200
    period = 14
    long = 26
    short = 12
    sig = 9
    r_df_set = get_upbit_df(coin_name, minute, count)
    r_df_set.rename(columns={'candle_date_time_kst':'Date', 'opening_price':'Open', 'high_price':'High','low_price':'Low', 'candle_acc_trade_volume':'Volume', 'trade_price':'Close'}, inplace=True)
    defulat_col = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    r_df_set = r_df_set[defulat_col]
    r_df_set.set_index('Date', inplace=True)
    trade_df = r_df_set[30:]
    index_df = pd.DataFrame()
    for i in range(0+1,count-30+1,1):
        #i = 171
        r_df = r_df_set[i:i+30]
        rsi, rmi, macd, signal, oscilator, bol_higher, bol_lower = get_index_set(r_df, 'Close', 'Open', 'Low', period, lookback, short, long, sig)
        temp_df = pd.DataFrame({'rsi':rsi, 'rmi':rmi,
                                 'macd':macd, 'signal':signal, 'oscilator':oscilator,
                                 'bol_higher':bol_higher, 'bol_lower':bol_lower}, index=[max(r_df.index)])
        index_df = pd.concat([index_df, temp_df], axis=0)
    df = pd.concat([trade_df,index_df], axis=1)

    index_col = ['Open','High','Low','Close','Volume','bol_higher','bol_lower','rsi','rmi','macd','signal','oscilator']
    index_c_col =  ['c_'+x for x in index_col]

    df_changed = df[index_col].apply(lambda x: x.diff(1), axis=0)
    df_changed.columns = index_c_col
    temp_df = pd.DataFrame()
    for x in index_col:
        ori_ocl = 'c_'+x
        new_col = 'r_'+x
        temp_df[new_col] = df_changed[ori_ocl]/df[x]
        temp_df[new_col] = temp_df[new_col].fillna(0)
        temp_df[new_col] = temp_df[new_col].apply(lambda x: np.sign(x) if abs(x) > 1 else x)

    res_df = pd.concat([df, df_changed, temp_df], axis=1)
    res_df.dropna(inplace=True)
    res_df['low_price'] = np.min(res_df['Low'])
    res_df['min_price'] = min(np.min(res_df['Open']), np.min(res_df['Close']))
    res_df['max_price'] = np.min(res_df['Open'])
    X, X_test, y = res_df[:-1], res_df[-1:],(res_df[1:]['Close']-res_df[1:]['Open'])
    #time_row = 1
    res_df = res_df.tail(time_row).reset_index(drop=True)
    return res_df, X, y, X_test
#
# def get_pred_xgboost(df,n):
#     result = list()
#     for i in np.arange(1,n+1,1):
#         y = df['trade_price'].diff(i)
#         x = df[['opening_price', 'high_price', 'low_price',
#            'candle_acc_trade_price', 'candle_acc_trade_volume', 'rsi', 'rmi',
#            'nvr', 'macd', 'signal', 'oscilator', 'bol_higher', 'bol_lower']]
#         X = x[:-i]
#         y = y[i:]
#         y[y>0] = 1
#         y[y<0] = 0
#         if i == 1:
#             predict_x = x[-i:]
#         else:
#             predict_x = x[-i:-i+1]
#         xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                                          colsample_bytree=1, max_depth=7)
#         scaler = StandardScaler()
#         scaler.fit(X)
#         X = scaler.transform(X)
#         r_sq = xgb_model.fit(X, y)
#         predict_x = scaler.transform(predict_x)
#         predictions = r_sq.predict(predict_x)
#         res = predictions[0]
#         result.append(res)
#     return result
#################################################
# RSI(Relative Strength Index)
#################################################
# RSI = RS / (1+RS) = AU / (AU+AD)

# Welles Wilder는 70% 이상을 초과매수 국면으로, 30% 이하를 초과매도 국면으로 규정했다.
# 전략 1. RSI가 70%를 넘어서면 매도 포지션을, 30% 밑으로 떨어지면 매수 포지션을 취하는 방식이 있다.
# 전략 2. RSI가 70%를 넘어선 후 머물러 있다가 다시 70%를 깨고 내려오면 매도를, RSI가 30% 밑으로 내려가 머물러 있다가 다시 30% 이상으로 올라오면 매수하는 방식
# 전략 3. RSI가 50%를 상향 돌파하면 매수, RSI가 50%를 하향 돌파하면 매도하는 식의 방법으로 매매할 수 있다.
import requests
import pandas as pd
import time

def get_rsi(df, col, period: int = 9):
    #ohlc = temp_df
    #df = r_df
    delta = df[col].diff()
    gains, declines = delta.copy(), delta.copy()
    gains[gains < 0] = 0
    declines[declines > 0] = 0
    _gain = gains.ewm(com=(period - 1), min_periods=period).mean()
    _loss = declines.abs().ewm(com=(period - 1), min_periods=period).mean()
    RS = _gain / _loss
    rsi = 100 - (100 / (1 + RS))
    res = (float(rsi[-1:]) + (period - 1) * float(rsi[-2:-1])) / period
    return res

def get_rmi(df, col, lookback: int=5, period:int = 9):
    delta = df[col].diff(lookback)
    momup, momdown = delta.copy(), delta.copy()
    momup[momup < 0] = 0
    momdown[momdown > 0] = 0
    # _momup = momup.ewm(com=(period - 1), min_periods=period).mean()
    # _momdown = momdown.abs().ewm(com=(period - 1), min_periods=period).mean()
    _momup = momup[-period:].mean()
    _momdown = momdown[-period:].abs().mean()
    if _momdown == 0:
        rm = 100
    else:
        rm = _momup / _momdown

    rmi = rm / (1 + rm)
    #RMI = _momup.mean() / (_momup.mean() + _momdown.mean())
    return rmi


def get_MACD(tradePrice, short:int = 12,long:int = 26, sig:int = 9):
    exp12 = tradePrice.ewm(span=short, adjust=False).mean()
    exp26 = tradePrice.ewm(span=long, adjust=False).mean()
    macd = exp12-exp26
    signal = macd.ewm(span=sig, adjust=False).mean()
    oscillator = macd - signal
    return float(macd[-1:]), float(signal[-1:]), float(oscillator[-1:])

def get_upbit_df(coin_name, minute, count):
    if minute == 'day':
        url = "https://api.upbit.com/v1/candles/days"
    elif minute == 'week':
        url = "https://api.upbit.com/v1/candles/weeks"
    else:
        url = "https://api.upbit.com/v1/candles/minutes/"+str(minute)
    #url = "https://api.upbit.com/v1/candles/minutes/1"
    querystring = {"market":coin_name,"count":str(30+count)}
    #querystring = {"market":coin_name,"count":str(200)}
    response = requests.request("GET", url, params=querystring)
    time.sleep(0.1)
    data = response.json()
    df = pd.DataFrame(data) # 최근것이 위에있다.
    r_df = df.reindex(index=df.index[::-1]).reset_index(drop=True)
    return r_df

def get_index_set(r_df, col, o_col, l_col, period, lookback, short,long,sig):
    rsi = get_rsi(r_df, col, period)
    rmi = get_rmi(r_df, col, lookback, period)*100
    macd, signal, oscilator = get_MACD(r_df[col], short, long, sig)
    bol_median, bol_higher, bol_lower = f_bol(r_df, col, o_col, l_col)
    return rsi, rmi, macd, signal, oscilator, bol_higher, bol_lower

def f_rsi(df, line):

    def get_f_rsi(temp_df, line):
        temp_df = temp_df.reset_index(drop=True)
        a = temp_df['close'] - temp_df['open']
        b = np.sum(a[a > 0])/line
        c = abs(np.sum(a[a < 0]))/line
        if (c == 0):
            rsi = 100
        else:
            rs = b / c
            rsi = 100 - 100 / (1 + rs)
        return rsi

    i_set = [0, 1, 2, 3]
    rsi_set = list()
    #for i in range(line):
    for i in i_set:
        no = line - i
        after = get_f_rsi(df[:no], no)
        if i > 0:
            before = get_f_rsi(df[no:], i)
        else:
            before = 0
        rsi = (after * no + before * i) / line
        rsi_set.append(rsi)
    rsi_mean = np.mean(rsi_set)
    rsi_std = np.std(rsi_set)
    return rsi_mean, rsi_std




#################################################
# MACD(Moving Average Convergence & Divergence)
#################################################
# MACD는 장단기 이동평균 선간의 차이를 이용하여 매매신호를 포착하려는 기법으로 제럴드 아펠(Gerald Appel)에 의해 개발되었다.
# 주가 추세의 힘, 방향성, 시간을 측정하기 위해 사용된다.
# MACD의 원리는 장기 이동평균선과 단기이동평균선이 서로 멀어지게 되면(divergence) 언젠가는 다시 가까워져(convergence) 어느 시점에서 서로 교차하게 된다는 성질을 이용하여 두 개의 이동평균선이 멀어지게 되는 가장 큰 시점을 찾고자 하는 것이다.

# 계산 공식 :MACD : 12일 지수이동평균 - 26일 지수이동평균
# 시그널 : MACD의 9일 지수이동평균
# 오실레이터 : MACD값 - 시그널값
#
# 전략 1. MACD가 양으로 증가하면 매수한다.
# 전략 2. MACD가 시그널을 골드크로스 하면 매수한다.
# 전략 3. MACD가 0선을 상향 돌파하면 매수한다.
# https://md2biz.tistory.com/397
# def f_macd(df):
#     macd = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True).dropna()
#     macd.columns = ['macd', 'signal', 'days_9']
#     macd['oscillator'] = macd['macd'] - macd['days_9']
#     value_macd = np.mean(macd['macd'][-5:])
#     value_oscil = np.mean(macd['oscillator'][-5:])
#     return value_macd, value_oscil

def f_bol(df, col, o_col, l_col):
    df = df[0:len(df)-1]
    value = (df[col]+df[o_col])/2
    bol_median = np.mean(value)
    bol_std = np.std(value)
    bol_higher = bol_median + 1.96 * bol_std
    bol_lower = bol_median - 1.96 * bol_std
    return bol_median, bol_higher, bol_lower

def f_macd(df):
    ema12 = f_ema(df, 12)[-18:]
    ema26 = f_ema(df, 26)[-18:]
    macd = [i - j for i, j in zip(ema12, ema26)][-9:]
    signal = f_macd_signal(macd, 9)[-9:]
    oscillator = [i - j for i, j in zip(macd, signal)]
    return macd, signal, oscillator
def f_ema(df, length):
    #df = df2
    #length = 12
    sdf = df[0:length].reset_index(drop=True)
    ema = round(np.mean(df.close[0:length]),0)
    n = np.count_nonzero(df.close.to_list())
    sdf = df[length:n-1]
    res = [ema]
    ls = list(sdf.close)
    for i in range(np.count_nonzero(ls)-1):
        ema = round(ls[i+1]*2/(length+1) + ema*(1-2/(length+1)),2)
        res = res + [ema]
    return res
def f_macd_signal(macd, length):
    length  = 9
    s = macd[0:length]
    signal = round(np.mean(s), 0)
    macd = macd[-9:]
    n = np.count_nonzero(macd)
    res = [signal]
    for i in range(np.count_nonzero(macd)):
        signal = round(macd[i] * 2 / (length + 1) + signal * (1 - 2 / (length + 1)), 3)
        res = res + [signal]
    return res
################################
# RMI(Relative Momentum Index)
################################
# RSI의 공식에서는 오늘의 종가와 어제의 종가를 비교한 수치가 사용되지만 RMI에서는 일정기간 이전의 종가와 오늘의 종가를 비교한 수치가 사용됩니다
# RMI지표는 RSI에 기간이라는 기간 변수를 하나 더 추가한 것으로 기본적인 해석은 RSI와 동일하게 이해 한 된다.
# 전략 1. RSI가 70%를 넘어서면 매도 포지션을, 30% 밑으로 떨어지면 매수 포지션을 취하는 방식이 있다.
# 전략 2. RSI가 70%를 넘어선 후 머물러 있다가 다시 70%를 깨고 내려오면 매도를, RSI가 30% 밑으로 내려가 머물러 있다가 다시 30% 이상으로 올라오면 매수하는 방식
# 전략 3. RSI가 50%를 상향 돌파하면 매수, RSI가 50%를 하향 돌파하면 매도하는 식의 방법으로 매매할 수 있다.

def f_rmi(df):
    def get_f_rmi(temp_df, line):
        a = temp_df['close'] - temp_df['open']
        b = np.sum(temp_df['value'][a > 0]) / line
        c = abs(np.sum(temp_df['value'][a < 0])) / line
        if (c == 0):
            rmi = 100
        else:
            rs = b / c
            rmi = 100 - 100 / (1 + rs)
        return rmi
    line = 14
    rmi_set = list()
    i_set = [0, 1, 2, 3]
    #for i in range(line):
    for i in i_set:
        no = line - i
        after = get_f_rmi(df[:no],no)
        if i > 0:
            before = get_f_rmi(df[no:],i)
        else:
            before = 0
        rmi = (after * no + before * i) / line
        rmi_set.append(rmi)
    rmi_mean = np.mean(rmi_set)
    rsi_std = np.std(rmi_set)
    return rmi_mean

