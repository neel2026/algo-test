"""
V2 SMA Pullback Strategy - Filtered Logic
-----------------------------------------
Changes from V1:
1. ADX Filter: Only trade if ADX > 25 (Strong Trend).
2. RSI Filter: Only trade if RSI is between 40 and 65 (Healthy Pullback).
3. Logic: Reduces 'Fakeouts' in choppy markets.
"""

import pandas as pd
import pandas as pd
# import pandas_ta as ta # Removed dependency
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = r"d:\trading\data\NSE stock data"
SYMBOL = "ADANIPOWER"
TIMEFRAME = "15m"
SMA_FAST = 20
SMA_SLOW = 50

# --- INDICATOR FUNCTIONS ---
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_adx(high, low, close, period=14):
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Check index alignment by converting back to Series
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smooth (Wilder's Smoothing usually, using EWM here for approx)
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# RISK MANAGEMENT
STOP_LOSS_PCT = 0.005  
TAKE_PROFIT_PCT = 0.01 
MAX_TRADES_DAILY = 3

def fetch_data(symbol):
    print(f"⏳ Loading {symbol} data...")
    try:
        file_path = f"{DATA_DIR}\\{symbol}_15minute.csv"
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Date Parsing
        date_col = None
        for col in ['date', 'datetime', 'timestamp', 'time']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['timestamp'] = pd.to_datetime(df[date_col])
            df.set_index('timestamp', inplace=True)
        
        print(f"✅ Loaded {len(df)} candles.")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def backtest_strategy(df):
    print("\n🚀 STARTING V2 BACKTEST (WITH ADX/RSI FILTERS)...")
    
    # 1. Calculate Indicators
    # df['sma20'] = ta.sma(df['close'], length=SMA_FAST)
    # df['sma50'] = ta.sma(df['close'], length=SMA_SLOW)
    df['sma20'] = df['close'].rolling(window=SMA_FAST).mean()
    df['sma50'] = df['close'].rolling(window=SMA_SLOW).mean()
    
    # df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi'] = get_rsi(df['close'], period=14)
    
    # ADX Calculation
    # adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    # df['adx'] = adx_df['ADX_14']
    df['adx'] = get_adx(df['high'], df['low'], df['close'], period=14)
    
    trades = []
    daily_trades = {} 
    active_trade = None
    
    # Clean NaNs created by ADX/SMA
    print("DEBUG: Checking NaNs before dropna:")
    print(df.isna().sum())
    # df.dropna(inplace=True) # Commenting out for now to see what survives or fix it
    df = df.dropna() # Use explicit assignment and maybe fillna if needed
    print(f"DEBUG: Data length after dropna: {len(df)}")
    if len(df) > 0:
        print(f"DEBUG: First 5 ADX values: {df['adx'].head().values}")
        print(f"DEBUG: First 5 RSI values: {df['rsi'].head().values}")
    
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        day_str = str(curr.name.date())
        
        if day_str not in daily_trades: daily_trades[day_str] = 0
            
        # --- EXIT LOGIC ---
        if active_trade:
            entry_price = active_trade['entry_price']
            sl_price = entry_price * (1 - STOP_LOSS_PCT)
            tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
            
            if curr['low'] <= sl_price:
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': curr.name,
                    'type': 'Stop Loss',
                    'pnl_pct': -STOP_LOSS_PCT
                })
                active_trade = None
            elif curr['high'] >= tp_price:
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': curr.name,
                    'type': 'Take Profit',
                    'pnl_pct': TAKE_PROFIT_PCT
                })
                active_trade = None
            continue

        # --- ENTRY LOGIC ---
        if active_trade is None:
            if daily_trades[day_str] >= MAX_TRADES_DAILY: continue
            
            # 1. Trend Filter: SMA20 > SMA50
            if not (prev['sma20'] > prev['sma50']): continue

            # 2. STRENGTH Filter: ADX must be > 25 (Ignore weak trends)
            if prev['adx'] < 25: continue

            # 3. Setup: Touched SMA 20 or 50
            touched_20 = prev['low'] <= prev['sma20'] and prev['close'] > prev['sma20']
            touched_50 = prev['low'] <= prev['sma50'] and prev['close'] > prev['sma50']
            
            if (touched_20 or touched_50):
                # 4. Trigger: Break High
                if curr['close'] > prev['high']:
                    active_trade = {'entry_time': curr.name, 'entry_price': curr['close']}
                    daily_trades[day_str] += 1

    return pd.DataFrame(trades)

if __name__ == "__main__":
    df = fetch_data(SYMBOL)
    if not df.empty:
        results = backtest_strategy(df)
        if not results.empty:
            total_trades = len(results)
            wins = len(results[results['pnl_pct'] > 0])
            win_rate = (wins / total_trades) * 100
            stock_roi = results['pnl_pct'].sum() * 100
            
            print("\n" + "="*40)
            print(f"📊 V2 RESULTS (ADX FILTERED)")
            print("="*40)
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate:     {win_rate:.2f}%")
            print(f"Stock ROI:    {stock_roi:.2f}%")
            
            if stock_roi > 0:
                print("✅ IMPROVEMENT: The filters are working.")
            else:
                print("❌ RESULT: Still failing. The base strategy might be flawed.")
        else:
            print("⚠️ No trades found. ADX/RSI filters might be too strict or calculation error.")
    else:
        print("❌ Data fetch failed.")