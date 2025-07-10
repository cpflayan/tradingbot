# === trader.py ===
from binance.client import Client
from binance.enums import *
import config

client = Client(config.API_KEY, config.API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'


def get_balance():
    usdt = [b for b in client.futures_account_balance() if b['asset'] == 'USDT']
    return float(usdt[0]['balance']) if usdt else 0

def place_order(position_size, entry_price):
    qty = round((get_balance() * position_size * config.LEVERAGE) / entry_price, 3)
    try:
        client.futures_change_leverage(symbol=config.SYMBOL, leverage=config.LEVERAGE)
        order = client.futures_create_order(
            symbol=config.SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=qty
        )
        print(f"✅ 成功下單：BUY {qty} @ {entry_price}")
        return True
    except Exception as e:
        print(f"❌ 下單失敗：{e}")
        return False
