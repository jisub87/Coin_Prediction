import numpy as np
import pandas as pd
import pandas_ta as ta
#import xgboost
#from sklearn.preprocessing import StandardScaler
import kakao_message
import kakao_message as kakao
import upbit_coin
import threading
import math
import time
import os
import json
import datetime
def search_check(df):
    check_set = list()
    check1 = (df['rsi'][0] > df['b_rsi'][0]) & (df['rsi'][0] < 30)
    check2_1 = (df['rmi'][0] < 30) & (df['rmi'][0] < df['b_rmi'][0]) & (df['macd'][0] < df['b_macd'][0]) & (df['signal'][0] < df['b_signal'][0])
    check2_2 = (df['rmi'][0] > 30) & (df['rmi'][0] > df['b_rmi'][0]) & (df['macd'][0] > df['b_macd'][0]) & (df['signal'][0] > df['b_signal'][0])
    check2 = check2_1 | check2_2
    check3 = (df['oscilator'][0] > df['b_oscilator'][0])
    check4 = (df['bol_lower'][0] < df['b_bol_lower'][0]) & (df['bol_higher'][0] > df['b_bol_higher'][0])
    check_set.append(check1)
    check_set.append(check2)
    check_set.append(check3)
    check_set.append(check4)
    return check_set

def updown_check_df(df):
    check1 = df['rsi'] > df['b_rsi']
    check2 = df['rmi'] > df['b_rmi']
    check3 = df['macd'] > df['b_macd']
    check4 = df['signal'] > df['b_signal']
    check5 = df['oscilator'] > df['b_oscilator']
    check6 = df['bol_lower'] > df['b_bol_lower']
    check7 = df['bol_higher'] > df['b_bol_higher']
    check8 = df['nvr'] > df['b_nvr']
    df['rsi_p'] = check1
    df['rmi_p'] = check2
    df['macd_p'] = check3
    df['signal_p'] = check4
    df['oscilator_p'] = check5
    df['bol_lower_p'] = check6
    df['bol_higher_p'] = check7
    df['nvr_p'] = check8
    return df

def buy_check_df(df):
    check1 = (df['rsi'] > df['b_rsi']) & (df['rsi'] < 30)
    check2_1 = (df['rmi'] < 30) & (df['rmi'] < df['b_rmi']) & (df['macd'] < df['b_macd']) & (
            df['signal'] < df['b_signal'])
    check2_2 = (df['rmi'] > 30) & (df['rmi'] > df['b_rmi']) & (df['macd'] > df['b_macd']) & (
            df['signal'] > df['b_signal'])
    check2 = check2_1 | check2_2
    check3 = (df['oscilator'] > df['b_oscilator'])
    check4 = (df['bol_lower'] < df['b_bol_lower']) & (df['bol_higher'] > df['b_bol_higher'])
    df['rule1'] = check1
    df['rule2'] = check2
    df['rule3'] = check3
    df['rule4'] = check4
    return df

def buy_check(df):
    check_set = list()
    check1 = (df['rsi'][0] > df['b_rsi'][0]) & (df['rsi'][0] < 30)
    check2_1 = (df['rmi'][0] < 30) & (df['rmi'][0] < df['b_rmi'][0]) & (df['macd'][0] < df['b_macd'][0]) & (
                df['signal'][0] < df['b_signal'][0])
    check2_2 = (df['rmi'][0] > 30) & (df['rmi'][0] > df['b_rmi'][0]) & (df['macd'][0] > df['b_macd'][0]) & (
                df['signal'][0] > df['b_signal'][0])
    check2 = check2_1 | check2_2
    check3 = (df['oscilator'][0] > df['b_oscilator'][0])
    check4 = (df['bol_lower'][0] < df['b_bol_lower'][0]) & (df['bol_higher'][0] > df['b_bol_higher'][0])
    check_set.append(check1)
    check_set.append(check2)
    check_set.append(check3)
    check_set.append(check4)
    return check_set

def sell_check(df):
    check_set = list()
    check1 = (df['rsi'][0] < df['b_rsi'][0])
    check2 = (df['oscilator'][0] < df['b_oscilator'][0])
    check3 = (df['macd'][0] > df['b_macd'][0])
    check4 = (df['signal'][0] > df['b_signal'][0])
    #check5 = (df['rsi'][0] > 70) | (df['rmi'][0] > 70) | (df['bol_higher'][0] <= df['currency'][0])
    #check6 = (df['trade_price'][0] < df['b_opening_price'][0])
    check_set.append(check1)
    check_set.append(check2)
    check_set.append(check3)
    check_set.append(check4)
    return check_set

        # count = 30
        # period = 10
        # lookback = 5
        # df, message = get_coin_info(coin_name, 1, count, period, lookback)
        # #print('구매1/'+message, str(round(df['rsi'][0],2)),str(round(df['oscilator'][0],2)))
        # #talk_result = kakao.talk_check('구매/'+message)
        # pred_buy_price = df['low_price'][0]
        # orderbook = pyupbit.get_orderbook(coin_name)
        # df_orderbook = pd.DataFrame(orderbook['orderbook_units'])
        # buy_price_set = df_orderbook['bid_price']
        # step = 1
        # price = float(buy_price_set[step:step+1])
        # upbit_coin.reservation_cancel(upbit, coin_name, './buy_list')
        # try:
        #     upbit_coin.buy_coin(upbit, coin_name, price, investment)
        # except:
        #     print('구매 에러',coin_name, price, investment)
def check_high_overheat(x):
    criteria_low = (x['rsi'] >= 70)
    x.loc[criteria_low, 'high_overheat'] = 1
    x.fillna(0, inplace=True)
    return x['high_overheat']
def check_low_overheat(x):
    criteria_low = ((x['rsi'] <= 30) & (x['rmi'] <= 10)) | ((x['rsi'] <= 30) & (x['rmi'] > 50))
    x.loc[criteria_low, 'low_overheat'] = 1
    x.fillna(0, inplace=True)
    return x['low_overheat']
def check_influence(x):
    criteria_buy_low = (x['c1_nvr'] > 1) & (x['c2_nvr'] > 0)
    criteria_sell_low = (x['c1_nvr'] < -1) & (x['c2_nvr'] < 0)
    x.loc[criteria_buy_low, 'influence'] = 1
    x.loc[criteria_sell_low, 'influence'] = -1
    x['influence'].fillna(0, inplace=True)
    return x['influence']
def check_changed(x):
    #x = data_set
    r_rsi = (x['c1_rsi'] >0 ) & (x['c2_rsi']>0)
    r_rmi = (x['c1_rmi']>0) & (x['c2_rmi']>0)
    r_oscilator = (x['c1_oscilator'] > 0) & (x['c2_oscilator'] > 0)
    r_macd = (x['c1_macd'] > 0) & (x['c2_macd'] > 0)
    r_signal = (x['c1_signal'] > 0) & (x['c2_signal'] > 0)

    x['r_rsi'] = r_rsi
    x['r_oscilator'] = r_oscilator
    x['r_macd'] = r_macd
    x['r_signal'] = r_signal
    x['r_rmi'] = r_rmi
    criteria_up_low = (r_rsi) & (r_oscilator)
    criteria_up_mid = criteria_up_low & (r_macd)
    criteria_up_high = criteria_up_mid & (r_signal)
    criteria_down_low = (r_rsi) & (r_oscilator)
    criteria_down_mid = criteria_down_low & (r_macd)
    criteria_down_high = criteria_down_mid & (r_signal)
    x.loc[criteria_up_low, 'changed_price'] = 1
    x.loc[criteria_up_mid, 'changed_price'] = 2
    x.loc[criteria_up_high, 'changed_price'] = 3
    x.loc[criteria_down_low, 'changed_price'] = -1
    x.loc[criteria_down_mid, 'changed_price'] = -2
    x.loc[criteria_down_high, 'changed_price'] = -3
    x['changed_price'].fillna(0, inplace=True)
    return x


def get_trade_list(res_dir_1, default, buy_view_list, minutes):
    names = [x.replace('.csv','').replace('view_','') for x in buy_view_list]
    view_default = pd.read_csv(os.path.join(res_dir_1, default[0]))
    col_names = [str(x)+'분' for x in minutes]
    default_df = view_default['coin_name']
    for name in default+buy_view_list:
        col = name.replace('.csv','').replace('view_','')
        view = pd.read_csv(os.path.join(res_dir_1,name))
        view['sum'] = view[col_names].sum(axis=1)
        view.loc[view['1분'] >= 1,'check'] = 1
        view['check'].fillna(0.5, inplace=True)
        view[col] = view['sum'] * view['check']
        view_df = view[['coin_name',col]]
        default_df = pd.merge(default_df,view_df, on='coin_name')
    default_df['score'] = (default_df[names[1]] + default_df[names[2]] + default_df[names[3]]) * default_df[names[0]]
    default_df['best_score'] = default_df['score'] * default_df['overheat']
    buy_list = default_df[default_df['score']>0][['coin_name','score']].sort_values('score', ascending=False).head(5)
    buy_list['침체'] = 'N'
    best_buy_list = default_df[default_df['best_score']>0][['coin_name','best_score']].sort_values('best_score', ascending=False).head(5)
    best_buy_list.rename(columns={'best_score':'score'}, inplace=True)
    best_buy_list['침체'] = 'Y'
    result = pd.concat([best_buy_list, buy_list], axis=0)
    result.rename(columns={'score':'점수', 'coin_name':'종목'}, inplace=True)
    #result = pd.DataFrame({'거래순위':['1','2'], 'coin_name':[', '.join(best_buy_list), ', '.join(buy_list)]})
    return result


def genderation_pivot_tot(data_set, minute):
    base = data_set[data_set['minute'] == minute][['Order', 'coin_name', 'changed_trade_price']]
    base.loc[base['changed_trade_price'] >= 0, 'sign'] = 1
    base.loc[base['changed_trade_price'] < 0, 'sign'] = -1

    res = base.pivot_table(index=['sign', 'coin_name'], columns='Order', values='changed_trade_price')
    res = res.reset_index().sort_values(['sign', 1], ascending=[False, False])

    plus_price = res[res['sign'] == 1][[1, 2, 3, 4, 5]].sum(axis=0)
    minus_price = res[res['sign'] == -1][[1, 2, 3, 4, 5]].sum(axis=0)

    RS = plus_price / abs(minus_price)
    rsi = 100 - (100 / (1 + RS))

    tot_price = res[[1, 2, 3, 4, 5]].sum(axis=0)
    summary_df = pd.DataFrame(
        {'tot_trade_volume': tot_price, 'tot_plus_volume': plus_price, 'tot_minus_volume': minus_price, 'rsi':round(rsi,2)})
    view_1 = summary_df.reset_index()
    view_1['tot_trade_volume'] = round(view_1['tot_trade_volume'], -6)/1000000
    view_1['tot_plus_volume'] = round(view_1['tot_plus_volume'], -6)/1000000
    view_1['tot_minus_volume'] = round(view_1['tot_minus_volume'], -6)/1000000

    merge_df = pd.merge(base, summary_df.reset_index(), on='Order')
    plus_df = merge_df[merge_df['sign'] == 1].reset_index(drop=True)
    plus_df['per'] = round(plus_df['changed_trade_price'] / plus_df['tot_plus_volume'] * 100, 2)
    plus_df['per'] = plus_df['per'].fillna(0)
    view_2 = plus_df.pivot_table(index='coin_name', columns='Order', values='per').sort_values(5,
                                                                                               ascending=False).reset_index()
    view_2 = view_2[view_2[5] > 0].fillna('')

    minus_df = merge_df[merge_df['sign'] == -1].reset_index(drop=True)
    minus_df['per'] = round(minus_df['changed_trade_price'] / minus_df['tot_minus_volume'] * 100, 2)
    minus_df['per'] = minus_df['per'].fillna(0)
    view_3 = minus_df.pivot_table(index='coin_name', columns='Order', values='per').sort_values(5,
                                                                                                ascending=False).reset_index()
    view_3 = view_3[view_3[5] > 0].fillna(0)
    return view_1, view_2, view_3

def update_info(minutes):
    print('start - data merge')
    o_path = 'C:/Users/admin/PycharmProjects/Flask_Web/data'
    dir_1 = os.path.join(o_path, 'coin_info_set')
    coin_name_set = os.listdir(dir_1)
    data_set = pd.DataFrame()
    for coin_name in coin_name_set:
        # coin_name = coin_name_set[0]
        dir_2 = os.path.join(dir_1, coin_name)
        for minute in minutes:
            #minute = 60
            dir_3 = os.path.join(dir_2, str(minute))
            if os.path.isdir(dir_3):
                data_path = os.listdir(dir_3)
                path = os.path.join(dir_3, data_path[0])
                if os.path.isfile(path):
                    try:
                        data = pd.read_csv(path)
                        data_set = pd.concat([data_set, data], axis=0)
                    except:
                        print('error:',coin_name, minute, data)
    data_set.reset_index(drop=True, inplace=True)
    data_set.iloc[:,3:] = round(data_set.iloc[:,3:],3)
    #print('end - data merge')
    # save - dataset
    res_dir_1 = os.path.join(o_path, 'coin_result_set')
    if not os.path.isdir(res_dir_1):
        os.makedirs(res_dir_1)
    data_set.to_csv(os.path.join(res_dir_1, 'tot_data.csv'), index=False)

    #print('start - data column filter')
    default_col = ['Date', 'minute', 'coin_name']
    index_col = ['Open', 'High', 'Low', 'Close', 'Volume', 'bol_higher', 'bol_lower',
                 'rsi', 'rmi', 'macd', 'signal', 'oscilator']

    # view 1: 분별 데이터 저장
    now = time.localtime()
    #file_name = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    file_name = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    #file_name = "%02d%02d%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
    pivot_names = ['price_tot', 'price_trend']
    data_set = pd.read_csv(os.path.join(res_dir_1, 'tot_data.csv'))
    data_set['changed_trade_price'] = (data_set['Close'] - data_set['Open']) * data_set['Volume']
    for minute in minutes:
        #minute = 1
        res_dir_2 = os.path.join(res_dir_1, 'view_'+str(minute))
        if not os.path.isdir(res_dir_2):
            os.makedirs(res_dir_2)

        # print('pivot1 - price')
        view_1, view_2, view_3 = genderation_pivot_tot(data_set, minute)
        view_2['coin_name'] = view_2['coin_name'].apply(lambda x: x.replace('KRW-',''))
        view_3['coin_name'] = view_3['coin_name'].apply(lambda x: x.replace('KRW-',''))
        view_1.to_csv(res_dir_2+'/price_tot.csv', index=False)
        view_2.to_csv(res_dir_2+'/price_plus.csv', index=False)
        view_3.to_csv(res_dir_2+'/price_minus.csv', index=False)

        #
        # # print('pivot2 - trend')
        # for idx, col in enumerate(index_col):
        #     res_dir_3 = os.path.join(res_dir_2, col)
        #     if not os.path.isdir(res_dir_3):
        #         os.makedirs(res_dir_3)
        #
    #print('end - pivot data')

    #minute = minutes[0]
    view_col = ['r_Volume','rsi', 'r_rsi', 'r_oscilator']
    for rcol in view_col:
        for minute in minutes:
            #col = index_col[0]
            res_dir_2 = os.path.join(res_dir_1, 'view_' + str(minute))
            if os.path.isdir(res_dir_2):
                #name = pivot_names[0]
                res_dir_3 = os.path.join(res_dir_2, rcol+'.csv')
                temp_df = data_set[data_set['minute']==minute][['Order','coin_name',rcol]]
                if rcol != 'rsi':
                    temp_df[rcol] = round(temp_df[rcol] * 100,2)
                else:
                    temp_df[rcol] = round(temp_df[rcol],2)

                view = temp_df.pivot_table(index='coin_name', columns='Order', values=rcol).reset_index().sort_values(5, ascending=False).reset_index(drop=True)
                view['coin_name'] = view['coin_name'].apply(lambda x: x.replace('KRW-', ''))
                view.to_csv(res_dir_3, index=False)

                #file_list = [int(x.replace('.csv', '')) for x in os.listdir(res_dir_3)]
                #file_list.sort(reverse=True)
                # file_list = ['%04d' % x for x in file_list]
                # if len(file_list) > 5:
                #     load_file = file_list[0:5]
                #     for remove_file in file_list[5:]:
                #         remove_path = os.path.join(res_dir_3, str(remove_file)+'.csv')
                #         try:
                #             os.remove(remove_path)
                #         except:
                #             print('제거error:',remove_path)
                # else:
                #     load_file = file_list
                #temp_df = pd.DataFrame()
                # for file in load_file:
                #     #file = file_list[0]
                #     data = pd.read_csv(os.path.join(res_dir_3, str(file)+'.csv'))
                #     data['time']=file
                #     temp_df = pd.concat([temp_df, data], axis=0)
                #last_data = temp_df.pivot_table(index='coin_name', columns='time', values='up_value')
                #temp_df.to_csv(os.path.join(res_dir_1, name+'_'+str(minute)+'.csv'), index=False)
    # print('end - pivot data')

def generation_coin_csv(coin_name, minute, count, period, lookback, time_row):
    path = 'C:/Users/admin/PycharmProjects/Flask_Web/data'
    df = get_coin_info(coin_name, minute, count, period, lookback, time_row)

    dir_1 = os.path.join(path, 'coin_info_set')
    dir_2 = os.path.join(dir_1, coin_name)
    dir_3 = os.path.join(dir_2, str(minute))
    for dir_d in [dir_1, dir_2, dir_3]:
        if not os.path.isdir(dir_d):
            os.makedirs(dir_d)
    data_path = os.path.join(dir_3, 'data.csv')
    df.to_csv(data_path, index=False)
    # if df['levels'][0] >= 4:
    #     message = '구매추천:%s, 상승강도:%s, 상승가속도:%s' % (df['coin_name'][0], df['levels'][0], df['bonus'][0])
    #     kakao.talk_check(message)


def get_tickers(tickers, minutes, count, period, lookback, time_row):
    print('코인 정보 생성: 시작')
    #tickers = search_tickers
    start = time.time()
    for coin_name in tickers:
        #coin_name = tickers[0]
        threads = list()
        for minute in minutes:
            #minute = minutes[0]
            threads = []
            th = threading.Thread(target=generation_coin_csv, args=(coin_name, minute, count, period, lookback, time_row))
            th.start()
            threads.append(th)
        for thread in threads:
            thread.join()
    end = time.time()
    print('코인 정보 생성: 종료', f"{end - start:.5f} sec")



def cutoff_check(upbit, coin_name, check_sell, currency, benefit, balance, cutoff_ratio):
    if (benefit < cutoff_ratio) & (check_sell>=1):
        print('예상을 벗어났다. 손절한다')
        print('손절1/', coin_name, benefit)
        upbit_coin.reservation_cancel(upbit, coin_name, './sell_list')
        upbit.sell_market_order(coin_name, balance)
        upbit_coin.reservation_cancel(upbit, coin_name, './sell_list')

def check_index_buy(res_df):
    criteria_1 = (res_df['rmi'][0] < 50) & (res_df['rsi'][0] < 40) # & (res_df['oscilator'][0] < 0)
    criteria_2 = criteria_1 & (res_df['r1_rmi'][0] > 0)
    criteria_3 = criteria_2 & (res_df['r1_oscilator'][0] > 0)
    criteria_4 = criteria_3 & (res_df['r1_rsi'][0] > 0)
    criteria_5 = criteria_4 & (res_df['r1_macd'][0] > 0)
    criteria_6 = criteria_5 & (res_df['r1_signal'][0] > 0)
    bonus = np.sum([res_df['r2_rsi'][0]>0, res_df['r2_rmi'][0]>0, res_df['r2_oscilator'][0]>0, res_df['r2_macd'][0]>0,res_df['r2_signal'][0]>0]) # max 3
    res_df['up_levels'] = 0
    if criteria_1:
        res_df['up_levels'] = 1
    if criteria_2:
        res_df['up_levels'] = 2
    if criteria_3:
        res_df['up_levels'] = 3
    if criteria_4:
        res_df['up_levels'] = 4
    if criteria_5:
        res_df['up_levels'] = 5
    if criteria_6:
        res_df['up_levels'] = 6
    res_df['up_bonus'] = bonus
    return res_df


def check_index_sell(res_df):
    criteria_1 = (res_df['rmi'][0] > 50) & (res_df['rsi'][0] > 40) & (res_df['oscilator'][0] > 0)
    criteria_2 = criteria_1 & (res_df['r1_oscilator'][0] < 0)
    criteria_3 = criteria_2 & (res_df['r1_rsi'][0] < 0)
    criteria_4 = criteria_3 & (res_df['r1_macd'][0] < 0)
    criteria_5 = criteria_4 & (res_df['r1_rmi'][0] < 0)
    criteria_6 = criteria_5 & (res_df['r1_signal'][0] < 0)

    bonus = np.sum(
        [res_df['r2_rsi'][0] < 0, res_df['r2_rmi'][0] < 0, res_df['r2_oscilator'][0] < 0, res_df['r2_macd'][0] < 0,
         res_df['r2_signal'][0] < 0])  # max 5

    res_df['down_levels'] = 0
    if criteria_1:
        res_df['down_levels'] = 1
    if criteria_2:
        res_df['down_levels'] = 2
    if criteria_3:
        res_df['down_levels'] = 3
    if criteria_4:
        res_df['down_levels'] = 4
    if criteria_5:
        res_df['down_levels'] = 5
    if criteria_6:
        res_df['down_levels'] = 6
    res_df['down_bonus'] = bonus
    return res_df

def check_index_influence(res_df): # 세력 평가 >0: 세력 가세
    criteria_1 = (res_df['nvr'][0] > 1)
    criteria_2 = (res_df['nvr'][0] > 1) & (res_df['r1_nvr'][0] > 0)
    criteria_3 = (res_df['nvr'][0] > 1) & (res_df['r1_nvr'][0] > 0) & (res_df['r2_nvr'][0] > 0)
    criteria_4 = (res_df['nvr'][0] < 1)
    criteria_5 = (res_df['nvr'][0] < 1) & (res_df['r1_nvr'][0] < 0)
    criteria_6 = (res_df['nvr'][0] < 1) & (res_df['r1_nvr'][0] < 0) & (res_df['r2_nvr'][0] < 0)
    res_df['influence'] = 0
    if criteria_1:
        res_df['influence'] = 1
    if criteria_2:
        res_df['influence'] = 2
    if criteria_3:
        res_df['influence'] = 3
    if criteria_4:
        res_df['influence'] = -1
    if criteria_5:
        res_df['influence'] = -2
    if criteria_6:
        res_df['influence'] = -3
    return res_df

def check_index_overheat(res_df): # 침체 평가
    criteria_1 = (res_df['rsi'][0] < 30) & (res_df['rmi'][0] < 10)
    criteria_2 = (res_df['rsi'][0] > 70) & (res_df['rmi'][0] > 90)
    if criteria_1:
        res_df['recession'] = 1
    else:
        res_df['recession'] = 0
    if criteria_2:
        res_df['overheat'] = 1
    else:
        res_df['overheat'] = 0
    return res_df

def check_price(res_df):
    res_df['changed_trade_price'] = res_df['Volume'][0] * res_df['Close'][0] * np.sign(res_df['Close'][0] - res_df['Open'][0])
    return res_df
'''
     금액얻기
'''
def get_price(coin_name, minute=1):
    line = count = 30
    period = 10
    lookback = 5
    time_row = 1
    r_df_set = get_upbit_df(coin_name, minute, count)

    period = 14
    long = 26
    short = 12
    sig = 9
    rsi, rmi, macd, signal, oscilator, bol_higher, bol_lower = get_index_set(r_df_set, 'trade_price', 'opening_price', 'low_price', period,
                                                                             lookback, short, long, sig)
    min_price = r_df_set['low_price'].mean() - 1.96 * r_df_set['low_price'].std()
    return min_price, rsi, rmi, macd, signal, oscilator, bol_higher, bol_lower
def get_sell_price(coin_name, minute=1):
    line = count = 30
    period = 10
    lookback = 5
    time_row = 1
    r_df_set = get_upbit_df(coin_name, minute, count)
    min_price = r_df_set['trade_price'].mean() + r_df_set['trade_price'].std()
    return min_price



import pyupbit
import my_index
def get_info_trade_coin(upbit, coin_name, minute):
    currency = pyupbit.get_current_price(coin_name)
    krw, my_balances = upbit_coin.get_my_balances(upbit)
    if len(my_balances[my_balances['coin_name'] == coin_name])>=1:
        my_avg_price = float(my_balances.loc[my_balances['coin_name'] == coin_name, 'avg_buy_price'])
        my_balance = float(my_balances.loc[my_balances['coin_name'] == coin_name, 'balance'])
    else:
        my_avg_price = currency
        my_balance = 0
    min_price, rsi, rmi, macd, signal, oscilator, bol_higher, bol_lower = my_index.get_price(coin_name, minute=minute)
    min_buy_price = min([min_price, my_avg_price * 0.99])
    min_buy_price = upbit_coin.round_price(min_buy_price)

    sell_price = my_index.get_sell_price(coin_name, minute=minute)
    min_sell_price = np.max([sell_price, my_avg_price * 1.01])
    min_sell_price = upbit_coin.round_price(min_sell_price)

    ratio = (currency-my_avg_price)/my_avg_price
    bol_ratio = (currency-bol_lower)/bol_lower

    return krw, my_avg_price, my_balance, min_buy_price, min_sell_price, rsi, rmi, macd, signal, oscilator, bol_higher, bol_ratio, ratio, currency

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

# def index_upbit(coin_name, minute):
#     # 154, 165
#     #minute = 1
#     k = 30
#     l = 165
#     i = l + k
#     #coin_name = 'KRW-BTC'
#     df = pd.DataFrame()
#     while len(df) <= 10:
#         url = "https://api.upbit.com/v1/candles/minutes/"+str(minute)
#         #url = "https://api.upbit.com/v1/candles/minutes/1"
#         querystring = {"market":coin_name,"count":str(i)}
#         #querystring = {"market":coin_name,"count":str(200)}
#         response = requests.request("GET", url, params=querystring)
#         data = response.json()
#         df = pd.DataFrame(data)
#         interval = 'minute'+str(minute)
#
#         if len(df)>=10:
#             break
#         else:
#             time.sleep(1)
#
#     df_rsi = df.reindex(index=df.index[::-1]).reset_index(drop=True) # 아래 최신
#     #df_rsi = df.reset_index(drop=True)
#     period = 10
#     lookback = 5
#     result_set = pd.DataFrame()
#     for j in range(k):
#         #j = 0
#         temp_df = df_rsi[j:j+period+1]
#         bol_median, bol_higher, bol_lower, env_higher, env_lower = f_bol(temp_df)
#         v9 = temp_df['candle_acc_trade_volume'][0:9].mean()
#         v12 = temp_df['candle_acc_trade_volume'][0:12].mean()
#         v26 = temp_df['candle_acc_trade_volume'][0:26].mean()
#         vr = round(v9 / np.mean(v12 + v26)*100,0)
#         index_rsi = get_rsi(temp_df, period).iloc[-1]
#         index_rmi = get_rmi(temp_df, lookback, period)
#         temp_df_2 = df_rsi[j:j+l]
#         macd, signal, oscillator = get_MACD(temp_df_2['trade_price'])
#         res = {'bol_higher':bol_higher,'env_lower':env_lower,'bol_lower':bol_lower,'vr':vr, 'rsi':index_rsi, 'rmi':index_rmi, 'macd':macd,'signal':signal, 'oscillator':oscillator}
#         res = pd.concat([temp_df_2[:1].reset_index(drop=True), pd.DataFrame(res,index=[0])], axis=1)
#         result_set = pd.concat([result_set,res], axis=0)
#     col = ['trade_price', 'high_price', 'low_price', 'bol_lower', 'env_lower',
#        'candle_acc_trade_price', 'candle_acc_trade_volume', 'vr', 'rsi',
#        'rmi', 'macd', 'signal', 'oscillator']
#     result_set = result_set[col].rename(columns={'candle_acc_trade_volume':'volume', 'trade_price':'close'})
#     for colum in result_set.columns:
#         result_set[colum] = result_set[colum].astype(float)
#
#     return result_set

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


################################
# MFI (Money Flow Index)
################################
# MFI는 주식 거래를 위한 자금의 유입과 유출 양을 측정하는 지표로, 추세 전환 시기를 예측하거나 시세 과열 또는 침체 정도를 파악하는데 유용하다.
# MFI는 RSI에 거래량 강도를 더한 지표로 이해할 수 있습니다.
# MFI가 20 이하이면 과매도, 80 이상이면 과매수 국면으로 이해할수 있습니다.

import requests
import pandas as pd
import json
#headers = {'Authorization': 'Bearer [YOUR_ACCESS_TOKEN]'}


def req(API_HOST, path,  method, data={}):
    url = API_HOST + path
    #print('HTTP Method: %s' % method)
    #print('Request URL: %s' % url)
    #print('Headers: %s' % headers)
    if method == 'GET':
        return requests.get(url)
        #return requests.get(url, headers=headers)
    else:
        return requests.post(url, data=data)
        #return requests.post(url, headers=headers, data=data)

def get_buy_list(API_HOST, HOST_root, method):
    response = req(API_HOST, HOST_root, method)
    #print("response status:\n%d" % response.status_code)
    # print("response headers:\n%s" % resp.headers)
    #print("response body:\n%s" % response.text)
    data = response.json()
    info = json.loads(data)
    df = pd.DataFrame(info)  # 최근것이 위에있다.
    return df
