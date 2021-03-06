'''
    평가 시스템 필요
'''
import pyupbit
import streamlit as st
import pandas as pd
import my_index
import modeling
import numpy as np

print("pandas version: ", pd.__version__)
pd.set_option('display.max_columns', 100)

''' 초기값'''
#tickers = pyupbit.get_tickers('KRW')

def coin_prediction(coin_name):
    minutes_res = list()
    for minute in [1, 3, 5, 10, 15, 30, 60, 240]:
        df, X, y, X_predict = my_index.get_coin_info(coin_name, minute)
        prediction = modeling.my_xgboost(X, y, X_predict, 0.1)
        res = {'x': minute, 'y': prediction[0]}
        minutes_res.append(res)
    return pd.DataFrame(minutes_res)

if __name__ == "__main__":
    #remove_coin = ['KRW-BTT', 'KRW-GMT']
    #ticekrs = [x for x in tickers if x not in remove_coin]
    tickers = pyupbit.get_tickers('KRW')
    st.title('Prediction Coin Price')
    res_df = pd.DataFrame()
    coin_name = st.sidebar.selectbox("Selection Coin", (tickers))

    if len(coin_name)>=1:
        st.header("Chart Data")
        count = 0
        res = pd.DataFrame()
        while True:
            if count ==0:
                data_load_state = st.text('Generation data:{}'.format(count))
            else:
                data_load_state = data_load_state.text('Generation data:{}'.format(count))
            df = coin_prediction(coin_name).reset_index(drop=True)
            df.set_index('x', inplace=True)
            res = pd.concat([res, df], axis=1).iloc[:,-10:]
            mean = res.mean(axis=1)
            std = res.std(axis=1)
            n = res.count(axis=1)
            lower = mean - 1.95 * std/np.sqrt(n)
            higher = mean + 1.95 * std/np.sqrt(n)
            plot_df = pd.concat([mean, lower, higher], axis=1)
            if count==0:
                element = st.line_chart(plot_df)
            else:
                element = element.line_chart(plot_df)
            count += 1
            data_load_state.text("Done:{}".format(count))