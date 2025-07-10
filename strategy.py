# === strategy.py ===
def load_model():
    import joblib
    clf = joblib.load("clf_model.pkl")
    reg = joblib.load("reg_model.pkl")
    return clf, reg

def extract_features(df):
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, CCIIndicator
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

def make_decision(clf, reg, df, config):
    df = extract_features(df)
    features = ['ema_gap', 'rsi', 'macd', 'cci', 'stoch', 'vol_ratio']
    x = df[features].dropna().iloc[-1:]
    prob = clf.predict_proba(x)[0, 1]
    pred_return = reg.predict(x)[0]
    if prob > config.THRESH_PROB and pred_return > config.THRESH_RET:
        position_size = min(pred_return * 100, config.MAX_POSITION)
        return True, position_size
    return False, 0.0
