# main.py - Forex AI Trading Bot for India (Kraken/Binance)

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot
from flask import Flask, jsonify, render_template_string
import threading
import time
import traceback 

# --- ML Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION LOADING ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# FOREX PAIRS - Use exchange-specific format
FOREX_PAIRS = os.getenv("FOREX_PAIRS", "EUR/USD,GBP/USD,USD/JPY,AUD/USD").split(',')
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d' 
ANALYSIS_INTERVAL = 30  # minutes

# Exchange Selection (choose one)
EXCHANGE_NAME = os.getenv("EXCHANGE", "kraken").lower()

# Initialize Bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Initialize Exchange based on selection
if EXCHANGE_NAME == "kraken":
    # Kraken supports forex and is available in India
    exchange = ccxt.kraken({
        'apiKey': os.getenv('KRAKEN_API_KEY', ''),
        'secret': os.getenv('KRAKEN_SECRET', ''),
        'enableRateLimit': True,
        'rateLimit': 1000,
    })
    print("✅ Using Kraken Exchange")
    
elif EXCHANGE_NAME == "binance":
    # Binance has some forex-like pairs (though technically crypto/stablecoin)
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY', ''),
        'secret': os.getenv('BINANCE_SECRET', ''),
        'enableRateLimit': True,
        'rateLimit': 1200,
    })
    print("✅ Using Binance Exchange")
    
elif EXCHANGE_NAME == "wazirx":
    # WazirX is India-based
    exchange = ccxt.wazirx({
        'apiKey': os.getenv('WAZIRX_API_KEY', ''),
        'secret': os.getenv('WAZIRX_SECRET', ''),
        'enableRateLimit': True,
        'rateLimit': 1000,
    })
    print("✅ Using WazirX Exchange")
    
elif EXCHANGE_NAME == "bybit":
    # Bybit supports forex CFDs
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY', ''),
        'secret': os.getenv('BYBIT_SECRET', ''),
        'enableRateLimit': True,
        'rateLimit': 1000,
    })
    print("✅ Using Bybit Exchange")
    
else:
    # Default to Kraken (no API key needed for public data)
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 1000,
    })
    print("✅ Using Kraken Exchange (default, no API key needed for signals)")

# Global ML Model and Scaler
ML_MODEL = None
SCALER = None

# ========== FLASK WEB SERVER & STATUS TRACKING ==========
app = Flask(__name__) 

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_pairs": FOREX_PAIRS,
    "uptime_start": datetime.now().isoformat(),
    "market_type": "FOREX",
    "exchange": EXCHANGE_NAME.upper()
}

@app.route('/')
def home():
    pairs_list = '<br>'.join([f"• {pair}" for pair in FOREX_PAIRS])
    return render_template_string(
        "<h1>🌍 Forex AI Trading Bot Status</h1>" + 
        f"<p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>" +
        f"<p><strong>Status:</strong> {bot_stats['status']}</p>" + 
        f"<p><strong>Total Analyses:</strong> {bot_stats['total_analyses']}</p>" +
        f"<p><strong>Last Analysis:</strong> {bot_stats['last_analysis'] or 'N/A'}</p>" +
        f"<p><strong>Monitored Pairs:</strong></p>{pairs_list}"
    )

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/status')
def status():
    return jsonify(bot_stats), 200

# ========== FOREX-SPECIFIC HELPERS ==========

def get_pip_value(pair):
    """Returns pip value based on the forex pair"""
    if 'JPY' in pair:
        return 0.01  # For JPY pairs, 1 pip = 0.01
    return 0.0001  # For other pairs, 1 pip = 0.0001

def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility measurement"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ========== ML TRAINING FUNCTION ==========

def train_prediction_model(df):
    """Trains a Logistic Regression model for forex prediction."""
    global SCALER
    
    try:
        if len(df) < 100:
            print(f"⚠️ Not enough data ({len(df)} rows, need 100+) for ML training. Skipping.")
            return None, None

        # 1. Target Definition
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        # 2. Feature Engineering
        df['fast_over_slow'] = np.where(df['fast_sma'] > df['slow_sma'], 1, 0)
        df['close_over_fast'] = np.where(df['close'] > df['fast_sma'], 1, 0)
        
        # ATR-based volatility
        df['atr'] = calculate_atr(df)
        df['volatility'] = df['atr'] / df['close']
        
        # Additional features
        df['momentum'] = df['close'].pct_change(5)
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Safety checks
        df['volatility'] = df['volatility'].replace([np.inf, -np.inf], 0).fillna(0)
        df['momentum'] = df['momentum'].replace([np.inf, -np.inf], 0).fillna(0)
        df['price_range'] = df['price_range'].replace([np.inf, -np.inf], 0).fillna(0)
        
        df['volatility'] = np.clip(df['volatility'], 0, 0.1)
        df['momentum'] = np.clip(df['momentum'], -0.1, 0.1)
        df['price_range'] = np.clip(df['price_range'], 0, 0.1)
        
        df = df.dropna()
        
        if len(df) < 50:
            print(f"⚠️ After cleaning, only {len(df)} rows left. Need 50+. Skipping ML.")
            return None, None
        
        X = df[['fast_over_slow', 'close_over_fast', 'volatility', 'momentum', 'price_range']].copy()
        y = df['target'].copy()
        
        split_idx = int(len(X) * 0.9)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        SCALER = StandardScaler()
        X_train_scaled = SCALER.fit_transform(X_train)

        model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_train_scaled, y_train)
        print(f"✅ ML Model trained successfully. Training Accuracy: {accuracy:.2%}")
        return model, SCALER
        
    except Exception as e:
        print(f"❌ ML Training Error: {e}")
        traceback.print_exc()
        return None, None

# ========== TECHNICAL ANALYSIS FUNCTIONS ==========

def calculate_forex_pivot_points(df_daily):
    """Calculates Forex Pivot Points (Standard method)"""
    if df_daily.empty or len(df_daily) < 2:
        return None

    prev_day = df_daily.iloc[-2]  
    H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    
    PP = (H + L + C) / 3.0
    
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    R3 = H + 2 * (PP - L)
    S3 = L - 2 * (H - PP)
    
    return {
        'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3
    }

def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=500):
    """Fetches forex data and prepares for analysis."""
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.dropna()
        
        # Technical indicators
        df['fast_sma'] = df['close'].rolling(window=9).mean()
        df['slow_sma'] = df['close'].rolling(window=20).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['rsi'] = calculate_rsi(df)
        df['atr'] = calculate_atr(df)
        
        df = df.dropna() 
        
        if len(df) < 20: 
            return pd.DataFrame(), None
        
        # Daily data for pivots
        ohlcv_daily = exchange.fetch_ohlcv(symbol, daily_timeframe, limit=20) 
        df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_daily.set_index('timestamp', inplace=True)
        
        pivot_levels = calculate_forex_pivot_points(df_daily)
        
        return df, pivot_levels
        
    except Exception as e:
        print(f"❌ Data fetch error for {symbol}: {e}")
        return pd.DataFrame(), None

def get_forex_signal(df, pivot_levels, pair):
    """Determines forex trading signal with ML prediction."""
    
    latest = df.iloc[-1]
    current_price = latest['close']
    fast_sma = latest['fast_sma']
    slow_sma = latest['slow_sma']
    rsi = latest['rsi']
    atr = latest['atr']
    
    pip_value = get_pip_value(pair)
    
    # --- ML Prediction ---
    ml_prediction = "NEUTRAL (No Model)"
    ml_confidence = "N/A"
    
    if ML_MODEL is not None and SCALER is not None:
        try:
            current_volatility = atr / current_price if current_price > 0 else 0
            current_volatility = np.clip(current_volatility, 0, 0.1)
            
            momentum = df['close'].pct_change(5).iloc[-1]
            momentum = np.clip(momentum, -0.1, 0.1) if not np.isnan(momentum) else 0
            
            price_range = (latest['high'] - latest['low']) / current_price
            price_range = np.clip(price_range, 0, 0.1)
            
            is_fast_over_slow = 1 if fast_sma > slow_sma else 0
            is_close_over_fast = 1 if current_price > fast_sma else 0
            
            latest_features = pd.DataFrame({
                'fast_over_slow': [is_fast_over_slow],
                'close_over_fast': [is_close_over_fast],
                'volatility': [current_volatility],
                'momentum': [momentum],
                'price_range': [price_range]
            })
            
            X_predict_scaled = SCALER.transform(latest_features)
            prediction = ML_MODEL.predict(X_predict_scaled)[0]
            probability = ML_MODEL.predict_proba(X_predict_scaled)[0]
            bullish_prob = probability[1]
            bearish_prob = probability[0]
            
            if prediction == 1 and bullish_prob > 0.55:
                ml_prediction = "BULLISH"
                ml_confidence = f"{bullish_prob*100:.0f}%"
            elif prediction == 0 and bearish_prob > 0.55:
                ml_prediction = "BEARISH"
                ml_confidence = f"{bearish_prob*100:.0f}%"
            else:
                ml_prediction = "NEUTRAL"
                ml_confidence = f"{max(bullish_prob, bearish_prob)*100:.0f}%"

        except Exception as e:
            print(f"❌ ML PREDICTION FAILED: {e}")
            ml_prediction = "NEUTRAL (Error)"
            ml_confidence = "Error"
    
    # --- Trend Assessment ---
    trend = "Neutral"
    trend_emoji = "🟡"
    if fast_sma > slow_sma:
        trend = "Uptrend"
        trend_emoji = "🟢"
    elif fast_sma < slow_sma:
        trend = "Downtrend"
        trend_emoji = "🔴"
    
    # --- RSI Analysis ---
    rsi_signal = "Neutral"
    if rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    
    # --- Pivot Analysis ---
    pp = pivot_levels.get('PP', 'N/A')
    proximity_msg = ""
    if pp != 'N/A':
        distance_pips = (current_price - pp) / pip_value
        if abs(distance_pips) < 10:
            proximity_msg = f"Price near Pivot Point (PP: {pp:.5f})"
        elif distance_pips > 0:
            proximity_msg = f"Price {distance_pips:.1f} pips above PP"
        else:
            proximity_msg = f"Price {abs(distance_pips):.1f} pips below PP"
    
    # --- Final Signal ---
    signal = "HOLD"
    signal_emoji = "🟡"
    
    if "BULLISH" in ml_prediction and current_price > pp and rsi < 70:
        signal = "BUY"
        signal_emoji = "🚀"
    elif "BEARISH" in ml_prediction and current_price < pp and rsi > 30:
        signal = "SELL"
        signal_emoji = "🔻"
    elif rsi > 75:
        signal = "TAKE PROFIT / SELL"
        signal_emoji = "⚠️"
    elif rsi < 25:
        signal = "POTENTIAL BUY"
        signal_emoji = "💎"
    
    return {
        'trend': trend, 'trend_emoji': trend_emoji,
        'proximity_msg': proximity_msg, 'signal': signal,
        'signal_emoji': signal_emoji, 'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence, 'rsi': rsi, 'rsi_signal': rsi_signal,
        'atr': atr, 'pip_value': pip_value
    }

# ========== ASYNC SCHEDULER FUNCTIONS ==========

async def generate_and_send_signal(pair):
    """Fetches data, runs analysis, and sends Telegram message."""
    
    try:
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing {pair}...")
        
        df, pivot_levels = fetch_and_prepare_data(pair, TIMEFRAME)
        
        if df.empty or pivot_levels is None:
            message = f"🚨 Data Error for {pair}. Cannot generate signal."
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            return

        result = get_forex_signal(df, pivot_levels, pair)
        current_price = df.iloc[-1]['close']
        
        # Determine decimal places based on price
        if current_price < 10:
            decimals = 5
        elif current_price < 100:
            decimals = 3
        else:
            decimals = 2
        
        pivot_text = (
            f"<b>Daily Pivot Levels:</b>\n"
            f"  • <b>PP:</b> <code>{pivot_levels['PP']:.{decimals}f}</code>\n"
            f"  • <b>R1/S1:</b> <code>{pivot_levels['R1']:.{decimals}f}</code> / <code>{pivot_levels['S1']:.{decimals}f}</code>\n"
            f"  • <b>R2/S2:</b> <code>{pivot_levels['R2']:.{decimals}f}</code> / <code>{pivot_levels['S2']:.{decimals}f}</code>\n"
        )
        
        message = (
            f"╔════════════════════════════════╗\n"
            f"  🌍 <b>FOREX AI SIGNAL</b>\n"
            f"╚════════════════════════════════╝\n\n"
            
            f"<b>{pair}</b> | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"<b>Exchange:</b> {EXCHANGE_NAME.upper()}\n\n"
            
            f"---{result['signal_emoji']} <b>SIGNAL: {result['signal']}</b> {result['signal_emoji']}---\n\n"
            
            f"<b>💱 Current Rate:</b> <code>{current_price:.{decimals}f}</code>\n"
            f"<b>⏰ Timeframe:</b> {TIMEFRAME}\n"
            
            f"\n<b>🤖 AI FORECAST</b>\n"
            f"• Prediction: {result['ml_prediction']}\n"
            f"• Confidence: {result['ml_confidence']}\n"
            
            f"\n<b>📊 TECHNICAL INDICATORS</b>\n"
            f"{result['trend_emoji']} Trend: {result['trend']}\n"
            f"📈 RSI(14): <code>{result['rsi']:.1f}</code> - {result['rsi_signal']}\n"
            f"📊 ATR: <code>{result['atr']:.{decimals}f}</code> ({result['atr']/result['pip_value']:.1f} pips)\n"
            f"{result['proximity_msg']}\n\n"
            
            f"{pivot_text}\n"
            
            f"----------------------------------------\n"
            f"<i></i>"
        )

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"
        
        print(f"✅ Signal sent: {pair} | {result['signal']} | ML: {result['ml_prediction']}")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Error for {pair}: {e}")
        print(error_trace)
        
        diagnostic_message = (
            f"❌ <b>ERROR: {pair}</b>\n\n"
            f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S UTC')}\n"
            f"<b>Error:</b> <code>{str(e)[:200]}</code>"
        )
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=diagnostic_message, parse_mode='HTML')


async def start_scheduler_loop():
    """Sets up scheduler and runs asyncio loop."""
    
    global ML_MODEL, SCALER

    print(f"\n⏳ Training ML Model for Forex on {EXCHANGE_NAME.upper()}...")
    try:
        ohlcv_train = exchange.fetch_ohlcv(FOREX_PAIRS[0].strip(), TIMEFRAME, limit=600)
        df_train = pd.DataFrame(ohlcv_train, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_train['close'] = pd.to_numeric(df_train['close'])
        
        df_train['fast_sma'] = df_train['close'].rolling(window=9).mean()
        df_train['slow_sma'] = df_train['close'].rolling(window=20).mean()
        df_train = df_train.dropna()
        
        ML_MODEL, SCALER = train_prediction_model(df_train)
        
        if ML_MODEL is None:
            print("⚠️ Running without ML predictions")
        
    except Exception as e:
        print(f"❌ ML Training Failed: {e}")
        ML_MODEL = None
        SCALER = None

    scheduler = AsyncIOScheduler()
    
    for pair in [p.strip() for p in FOREX_PAIRS]:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[pair]) 
    
    scheduler.start()
    print(f"🚀 Forex Bot started on {EXCHANGE_NAME.upper()}. Signals every 30 minutes.")

    # Initial analysis
    for pair in [p.strip() for p in FOREX_PAIRS]:
        await generate_and_send_signal(pair)
        await asyncio.sleep(5)

    while True:
        await asyncio.sleep(60)


def start_asyncio_thread():
    """Background thread target."""
    try:
        asyncio.run(start_scheduler_loop())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()

scheduler_thread = threading.Thread(target=start_asyncio_thread, daemon=True)
scheduler_thread.start()

print(f"✅ Forex AI Bot initialized with Flask + Scheduler using {EXCHANGE_NAME.upper()}")

