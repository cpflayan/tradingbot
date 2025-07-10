# === config.py: 策略參數設置 ===
LOOKBACK = 300              # 用來計算技術指標的歷史資料長度
ROLLING_N = 10              # 用來計算最近 N 策略表現
FUTURE_N = 8                # 預測持倉時間
THRESH_PROB = 0.7           # XGBoost 分類信心門檻
THRESH_RETURN = 0.004       # 預期報酬門檻
MAX_POSITION = 0.2          # 最大動態倉位比例
MAX_SINGLE_LOSS = 0.01      # 單筆最大可容忍虧損
DRAWDOWN_THRESHOLD = 0.1    # 資金回撤風控
COOLDOWN_PERIOD = 10        # 回撤後冷卻期
LEVERAGE = 20               # 使用槓桿倍數
