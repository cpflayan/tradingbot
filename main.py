# === main.py ===
import time
import pandas as pd
import requests
import config
from strategy import load_model, make_decision
from trader import place_order

clf, reg = load_model()

while True:
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={config.SYMBOL}&interval=5m&limit=50"
        klines = requests.get(url).json()
        df = pd.DataFrame(klines, columns=["timestamp","open","high","low","close","volume","c1","c2","c3","c4","c5","c6"])
        df = df[['timestamp','open','high','low','close','volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        decision, pos_size = make_decision(clf, reg, df, config)
        if decision:
            latest_price = float(df['close'].iloc[-1])
            place_order(pos_size, latest_price)

    except Exception as e:
        print(f"⚠️ 錯誤：{e}")

    time.sleep(60)  # 每分鐘檢查一次
