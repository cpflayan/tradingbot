from binance.client import Client
from binance.enums import *
from config import *
from strategy import fetch_klines, load_models, get_signal
import time

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

def place_order(side, quantity):
    try:
        client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        order = client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"✅ 成功下單：{side} {quantity}")
        return order
    except Exception as e:
        print(f"❌ 下單失敗：{e}")
        return None

def run():
    df = fetch_klines(client)
    clf, reg = load_models()
    proba, ret_pred = get_signal(df, clf, reg)

    if proba > 0.7 and ret_pred > 0.004:
        usdt_balance = float(client.futures_account_balance()[6]['balance'])
        close_price = df['close'].iloc[-1]
        qty = round((usdt_balance * 0.2 * LEVERAGE) / close_price, 3)
        place_order("BUY", qty)
    else:
        print(f"📊 無進場條件: proba={proba:.2f}, pred_ret={ret_pred:.4f}")

if __name__ == "__main__":
    run()
