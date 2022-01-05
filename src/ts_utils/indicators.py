
import subprocess
import sys

import pandas as pd

# installation of package extras
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# suggested by:
# https://stackoverflow.com/questions/49648391/how-to-install-ta-lib-in-google-colab/49660479
install('talib-binary')
import talib

def add_indicators(df):
    '''
    Function that adds: EMA, ROC, MOM, RSI, STOK, STOD, ULTOSC, ADX
    '''
    # Add Exponential Moving Average
    def EMA(df, n):
        EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name="EMA_" + str(n))
        return EMA

    df['EMA10'] = EMA(df, 10)
    df['EMA50'] = EMA(df, 50)
    df['EMA200'] = EMA(df, 200)

    # Rate of Change
    df['ROC10'] = talib.ROC(df['Close'], 10)
    df['ROC30'] = talib.ROC(df['Close'], 30)

    # Momentum
    df['MOM10'] = talib.MOM(df['Close'], 10)
    df['MOM30'] = talib.MOM(df['Close'], 30)

    # Relative Strength index
    df['RSI10'] = talib.RSI(df['Close'], timeperiod=10)
    df['RSI30'] = talib.RSI(df['Close'], timeperiod=30)
    df['RSI200'] = talib.RSI(df['Close'], timeperiod=200)

    # Stochastics
    df['%K10'], df['%D10'] = talib.STOCH(df['High'],
                                         df['Low'],
                                         df['Close'],
                                         fastk_period=10,
                                         slowk_period=3,
                                         slowk_matype=0,
                                         slowd_period=3,
                                         slowd_matype=0)

    df['%K30'], df['%D30'] = talib.STOCH(df['High'],
                                         df['Low'],
                                         df['Close'],
                                         fastk_period=30,
                                         slowk_period=3,
                                         slowk_matype=0,
                                         slowd_period=3,
                                         slowd_matype=0)


    df['%K200'], df['%D200'] = talib.STOCH(df['High'],
                                           df['Low'],
                                           df['Close'],
                                           fastk_period=200,
                                           slowk_period=3,
                                           slowk_matype=0,
                                           slowd_period=3,
                                           slowd_matype=0)
    # ULTOSC
    df['ULTOSC'] = talib.ULTOSC(df['High'],
                            df['Low'],
                            df['Close'],
                            timeperiod1=7,
                            timeperiod2=14,
                            timeperiod3=28)

    # Average Dirctional Index
    df['ADX7'] = talib.ADX(df['High'],
                           df['Low'],
                           df['Close'],
                           timeperiod=7)

    df['ADX14'] = talib.ADX(df['High'],
                            df['Low'],
                            df['Close'],
                            timeperiod=14)

    df['ADX21'] = talib.ADX(df['High'],
                            df['Low'],
                            df['Close'],
                            timeperiod=21)
    return df



