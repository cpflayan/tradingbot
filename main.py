# === main.py: 即時交易主程式 ===
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from xgboost import XGBClassifier, XGBRegressor
from binance.client import Client
from binance.enums import *
from config import *

API_KEY = "你的API"
API_SECRET = "你的SECRET"
client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

symbol = "ETHUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE

model_clf = XGBClassifier()
model_reg = XGBRegressor()
model_clf.load_model("clf_model.json")
model_reg.load_model("reg_model.json")

def fetch_klines(symbol, interval, lookback):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume',
        'taker_buy_quote_volume', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    return df

def compute_features(df):
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

def place_order(side, quantity):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"\n✅ 成功下單：{side} {quantity} {symbol}")
        return order
    except Exception as e:
        print(f"❌ 下單失敗：{e}")
        return None

# === 主邏輯 ===
while True:
    try:
        df = fetch_klines(symbol, interval, LOOKBACK)
        df = compute_features(df)
        features = ['ema_gap', 'rsi', 'macd', 'cci', 'stoch', 'vol_ratio']
        row = df.dropna().iloc[-1:]
        X_pred = row[features]
        prob = model_clf.predict_proba(X_pred)[0][1]
        pred_ret = model_reg.predict(X_pred)[0]

        print(f"{datetime.utcnow()} - proba: {prob:.3f}, pred_return: {pred_ret:.4f}")

        if prob > THRESH_PROB and pred_ret > THRESH_RETURN:
            price = df['close'].iloc[-1]
            usdt_balance = float(client.futures_account_balance()[6]['balance'])
            qty = round((usdt_balance * MAX_POSITION * LEVERAGE) / price, 3)
            place_order("BUY", qty)

        time.sleep(60)  # 每分鐘跑一次

    except Exception as e:
        print(f"錯誤：{e}")
        time.sleep(30)
        continue
