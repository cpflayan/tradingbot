# === strategy.py ===

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
import joblib
from config import *

def fetch_klines(client):
    klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df

def apply_indicators(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema_gap'] = df['ema20'] - df['ema50']
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    return df

def load_models():
    clf = joblib.load(MODEL_CLF)
    reg = joblib.load(MODEL_REG)
    return clf, reg

def get_signal(df, clf, reg):
    df = df.copy()
    df = apply_indicators(df).dropna()
    latest = df.iloc[-1:]
    features = ['ema_gap', 'rsi', 'macd', 'cci', 'stoch', 'vol_ratio']
    proba = clf.predict_proba(latest[features])[:, 1][0]
    ret_pred = reg.predict(latest[features])[0]
    return proba, ret_pred
