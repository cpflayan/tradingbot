# 強化版 AI 自動交易策略（含 Debug 訊息）
from binance.client import Client
from binance.enums import *

API_KEY = "你的API_KEY"
API_SECRET = "你的API_SECRET"

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'  # 切換到 Testnet
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, CCIIndicator
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
def place_order(symbol, side, quantity, leverage=20):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"✅ 成功下單：{side} {quantity} {symbol}")
        return order
    except Exception as e:
        print(f"❌ 下單失敗：{e}")
        return None

# === 1. 讀取資料 ===
df = pd.read_csv("eth_usdt.csv")
df.columns = df.columns.str.lower().str.strip()
df['date'] = pd.to_datetime(df['timestamp'])
df.set_index('date', inplace=True)

# === 2. 技術指標 ===
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema_gap'] = df['ema20'] - df['ema50']
df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
df['macd'] = MACD(df['close']).macd_diff()
df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
df['vol_ma'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma']

# === 3. 預測標的 ===
future_n = 8
df['future_return'] = df['close'].shift(-future_n) / df['close'] - 1
df['target'] = (df['future_return'] > 0.004).astype(int)

# === 4. 清理資料 ===
features = ['ema_gap', 'rsi', 'macd', 'cci', 'stoch', 'vol_ratio']
df.dropna(subset=features + ['target'], inplace=True)
X = df[features]
y_class = df['target']
y_regr = df['future_return']
X_train, X_test = X.iloc[:-1000], X.iloc[-1000:]
yc_train, yc_test = y_class.iloc[:-1000], y_class.iloc[-1000:]
yr_train, yr_test = y_regr.iloc[:-1000], y_regr.iloc[-1000:]

# === 5. 模型訓練 ===
clf = XGBClassifier(n_estimators=200, max_depth=4, scale_pos_weight=3)
clf.fit(X_train, yc_train)
reg = XGBRegressor(n_estimators=200, max_depth=4)
reg.fit(X_train, yr_train)

# === 6. 預測與交易判斷 ===
proba = clf.predict_proba(X_test)[:, 1]
pred_return = reg.predict(X_test)

# === 7. 計算資金曲線 ===
thresh_prob = 0.7
thresh_ret = 0.004
sig = pd.Series(((proba > thresh_prob) & (pred_return > thresh_ret)).astype(int), index=X_test.index)
filt_return = pred_return.copy()
filt_return[(pred_return < 0.003) & (proba < 0.8)] = 0
MAX_POSITION = 0.2
pos_size = np.clip(filt_return * 100, 0, MAX_POSITION) * sig
future_returns = df.loc[X_test.index, 'future_return'].fillna(0).values
pos_size_arr = pos_size.values if isinstance(pos_size, pd.Series) else pos_size

# === 動態表現過濾 ===
lookback_n = 5
recent_returns = pd.Series(future_returns * pos_size_arr, index=X_test.index)
rolling_perf = recent_returns.rolling(lookback_n).mean()
sig[(rolling_perf < 0) & (sig == 1)] = 0


# === 資金動態模擬 ===
capital = initial_capital = 1.0
capital_high = capital
cooldown = 0
drawdown_threshold = 0.1
MAX_SINGLE_LOSS = 0.01
equity_dyn = []

# 假設持倉期分 8 天日報酬
daily_ret_matrix = (np.tile(future_returns.reshape(-1, 1), (1, future_n))) / future_n

for i in range(len(future_returns)):
    if capital < capital_high * (1 - drawdown_threshold):
        cooldown = 10
        capital_high = capital

    if cooldown > 0:
        equity_dyn.append(capital)
        cooldown -= 1
        continue

    if sig.iloc[i]:
        position = pos_size_arr[i]
        trade_equity = capital
        partial_exit = False
        holding = True

        if i == len(future_returns) - 1:  # 最新一筆才真實下單
           symbol = "ETHUSDT"
           usdt_balance = float(client.futures_account_balance()[6]['balance'])
           order_qty = round((usdt_balance * position * leverage) / df.loc[X_test.index[i], 'close'], 3)
           place_order(symbol, "BUY", order_qty)

        for day in range(future_n):
            daily_ret = daily_ret_matrix[i, day]
            pnl = position * daily_ret
            trade_equity *= (1 + pnl)
            change = (trade_equity / capital) - 1

            if change <= -MAX_SINGLE_LOSS:
                capital *= (1 - MAX_SINGLE_LOSS)
                holding = False
                break
            elif change >= 0.05:
                if not partial_exit:
                    capital *= (1 + position * daily_ret * 0.5)
                    position *= 0.5
                    partial_exit = True
                else:
                    capital *= (1 + position * daily_ret)
                    capital_high = max(capital_high, capital)
                    holding = False
                    break
            else:
                capital = trade_equity

        if holding:
            equity_dyn.append(capital)
    else:
        equity_dyn.append(capital)

equity_dyn = pd.Series(equity_dyn, index=X_test.index[:len(equity_dyn)])
strategy_returns = equity_dyn.pct_change().fillna(0)

# === Debug 日誌 ===
print("📊 模型預測總進場機會：", ((proba > thresh_prob) & (pred_return > thresh_ret)).sum())
print("📉 被 rolling_perf 過濾的筆數：", (rolling_perf < 0).sum())
print("📈 最終進場次數：", sig.sum())

# === 8. 圖表 ===
plt.figure(figsize=(12, 5))
plt.plot(equity_dyn, label="策略資金曲線")
plt.grid(True)
plt.legend()
plt.title("強化版 AI 策略資金曲線")
plt.xlabel("時間")
plt.ylabel("資金倍率")
plt.show()

# === 9. 評估 ===
valid_returns = strategy_returns[sig.values[:len(strategy_returns)] == 1]
win_rate = (valid_returns > 0).sum() / max(len(valid_returns), 1)
max_dd = ((equity_dyn.cummax() - equity_dyn) / equity_dyn.cummax()).max()

print(f"✅ 總報酬：{equity_dyn.iloc[-1]-1:.2%}")
print(f"📊 交易次數：{len(valid_returns)}")
print(f"✅ 勝率：{win_rate:.2%}")
print(f"📉 最大回撤：{max_dd:.2%}")
