# main.py - FINAL, STABLE KUCOIN PROXY WEB SERVICE (No API Key Required)

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler # CRITICAL: Switched to BackgroundScheduler for Gunicorn stability
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

# --- KUCOIN PROXY CONFIGURATION ---
EXCHANGE_ID = "kucoin" 
# Mapped Forex pairs to KuCoin's stablecoin crypto pairs (proxies for EUR/USD, GBP/USD, etc.)
SYMBOLS = os.getenv("FOREX_PROXIES", "EUR/USDT,BTC/USDT,ETH/USDT,XRP/USDT,SOL/USDT").split(',') 
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d'

# Initialize Bot and Exchange (Using KuCoin public access - No keys needed for market data)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

exchange_config = {
    'enableRateLimit': True,
    'rateLimit': 1000, 
    # NO API KEY OR SECRET REQUIRED
}

try:
    exchange = getattr(ccxt, EXCHANGE_ID)(exchange_config)
    exchange.load_markets() 
    print(f"✅ {EXCHANGE_ID.upper()} markets loaded successfully (Proxy mode).")
except Exception as e:
    print(f"❌ CRITICAL: Failed to initialize exchange: {e}")
    exit(1)

# Global ML Model and Scaler 
ML_MODEL = None
SCALER = None

# --- Status tracking variables ---
bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": SYMBOLS,
    "uptime_start": datetime.now().isoformat(),
    "exchange": EXCHANGE_ID.upper() + " (PROXY)"
}


# =========================================================================
# === SECTION 1: ALL FUNCTION DEFINITIONS (CRITICALLY IMPORTANT PLACEMENT) ===
# =========================================================================
# Note: Functions are simplified for proxy data and structured exactly as the working model.

def train_prediction_model(df):
    """Trains a Logistic Regression model and returns the model and scaler."""
    global SCALER
    if len(df) < 500: return None, None
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df['fast_over_slow'] = np.where(df['fast_sma'] > df['slow_sma'], 1, 0)
    df['close_over_fast'] = np.where(df['close'] > df['fast_sma'], 1, 0)
    df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0) 
    df = df.dropna()
    X = df[['fast_over_slow', 'close_over_fast', 'volatility']]; y = df['target']
    X_train = X.iloc[:-int(len(X) * 0.1)]; y_train = y.iloc[:-int(len(y) * 0.1)]
    SCALER = StandardScaler(); X_train_scaled = SCALER.fit_transform(X_train)
    model = LogisticRegression(solver='liblinear'); model.fit(X_train_scaled, y_train)
    print(f"✅ ML Model trained successfully. Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    return model, SCALER


def calculate_cpr_levels(df_daily):
    """Calculates Daily Pivot Points (PP, TC, BC, R/S levels)."""
    if df_daily.empty or len(df_daily) < 2: return None
    prev_day = df_daily.iloc[-2]; H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    PP = (H + L + C) / 3.0; BC = (H + L) / 2.0; TC = PP - BC + PP
    R1 = 2 * PP - L; S1 = 2 * PP - H; R2 = PP + (H - L); S2 = PP - (H - L); R3 = H + 2 * (PP - L); S3 = L - 2 * (H - PP)
    return {'PP': PP, 'TC': TC, 'BC': BC, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}


def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=500):
    """Fetches main chart data, prepares for analysis."""
    
    # KuCoin specific fix: use the symbol ID from markets, just like the working code.
    try:
        kucoin_symbol_id = exchange.markets[symbol]['id']
    except KeyError:
        print(f"Error: Symbol {symbol} not found in KuCoin market list. Check symbols.")
        return pd.DataFrame(), None

    ohlcv = exchange.fetch_ohlcv(kucoin_symbol_id, timeframe, limit=limit); df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True); df = df.dropna()
    df['fast_sma'] = df['close'].rolling(window=9).mean(); df['slow_sma'] = df['close'].rolling(window=20).mean(); df = df.dropna() 
    if len(df) < 20: return pd.DataFrame(), None
    
    ohlcv_daily = exchange.fetch_ohlcv(kucoin_symbol_id, daily_timeframe, limit=20) 
    df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df_daily.set_index('timestamp', inplace=True)
    cpr_levels = calculate_cpr_levels(df_daily)
    return df, cpr_levels


def get_trend_and_signal(df, cpr_levels):
    """Determines trend via SMA crossover and incorporates ML prediction."""
    global ML_MODEL, SCALER
    latest = df.iloc[-1]; current_price = latest['close']; fast_sma = latest['fast_sma']; slow_sma = latest['slow_sma']; ml_prediction = "NEUTRAL (No Model)"
    
    if ML_MODEL is not None and SCALER is not None:
        try:
            is_fast_over_slow = 1 if fast_sma > slow_sma else 0; is_close_over_fast = 1 if current_price > fast_sma else 0
            close_prices_recent = df['close'].iloc[-20:] 
            if len(close_prices_recent) < 20: current_volatility = 0.0
            else:
                 returns = close_prices_recent.pct_change().dropna(); current_volatility = returns.std(skipna=True).fillna(0)
                 current_volatility = current_volatility.iloc[-1] if isinstance(current_volatility, pd.Series) and not current_volatility.empty else float(current_volatility)
                 current_volatility = 0.0 if np.isinf(current_volatility) or np.isnan(current_volatility) else current_volatility

            latest_features = pd.DataFrame({'fast_over_slow': [is_fast_over_slow], 'close_over_fast': [is_close_over_fast], 'volatility': [current_volatility] }); X_predict_scaled = SCALER.transform(latest_features)
            prediction = ML_MODEL.predict(X_predict_scaled)[0]; probability = ML_MODEL.predict_proba(X_predict_scaled)[0]; bullish_prob = probability[1]
            
            if prediction == 1 and bullish_prob > 0.55: ml_prediction = f"BULLISH ({bullish_prob*100:.0f}%)"
            elif prediction == 0 and probability[0] > 0.55: ml_prediction = f"BEARISH ({probability[0]*100:.0f}%)"
            else: ml_prediction = "NEUTRAL (Low Conviction)"
        except Exception as e:
            ml_prediction = "NEUTRAL (ML Error)"
            
    trend = "Neutral"; trend_emoji = "🟡"
    if fast_sma > slow_sma: trend = "Uptrend"; trend_emoji = "🟢"
    elif fast_sma < slow_sma: trend = "Downtrend"; trend_emoji = "🔴"

    pp = cpr_levels.get('PP', 'N/A'); proximity_msg = ""; 
    price_format = ".4f" 
    if pp != 'N/A':
        distance_to_pp = current_price - pp
        if abs(distance_to_pp / pp) < 0.005: proximity_msg = "Price is near the <b>Central Pivot Point (PP)</b>."
        elif distance_to_pp > 0: proximity_msg = f"Price is <b>Above PP</b> ({pp:{price_format}})."
        else: proximity_msg = f"Price is <b>Below PP</b> ({pp:{price_format}})."
            
    signal = "HOLD"; signal_emoji = "🟡"
    if "BULLISH" in ml_prediction and current_price > pp: signal = "STRONG BUY"; signal_emoji = "🚀"
    elif "BEARISH" in ml_prediction and current_price < pp: signal = "STRONG SELL"; signal_emoji = "🔻"
        
    return trend, trend_emoji, proximity_msg, signal, signal_emoji, ml_prediction, "N/A" 


def generate_and_send_signal(symbol):
    """The main job executed by the scheduler."""
    
    try:
        # Uses a nested function to handle the async call from the sync scheduler thread
        async def send_message_async(text, parse_mode):
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode=parse_mode)

        df, cpr_levels = fetch_and_prepare_data(symbol, TIMEFRAME)
        
        if df is None or df.empty or cpr_levels is None:
            message = f"🚨 Data Fetch/Processing Error for {symbol} (Insufficient clean data)."
            asyncio.run(send_message_async(message, None))
            return

        trend, trend_emoji, proximity_msg, signal, signal_emoji, ml_prediction, ml_confidence = get_trend_and_signal(df, cpr_levels)
        current_price = df.iloc[-1]['close']; price_format = ".4f"
        
        cpr_text = (f"<b>Daily CPR Levels:</b>\n" f"  - <b>PP (Pivot Point):</b> <code>{cpr_levels['PP']:{price_format}}</code>\n" f"  - <b>R1/S1:</b> <code>{cpr_levels['R1']:{price_format}}</code> / <code>{cpr_levels['S1']:{price_format}}</code>\n" f"  - <b>R2/S2:</b> <code>{cpr_levels['R2']:{price_format}}</code> / <code>{cpr_levels['S2']:{price_format}}</code>\n")
        
        message = (
            f"╔════════════════════════════════╗\n" f"  🧠 <b>FOREX PROXY INTELLIGENCE REPORT</b>\n" f"╚════════════════════════════════╝\n\n"
            f"<b>{symbol}</b> (KuCoin Proxy) | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"---🚨 <b>{signal_emoji} FINAL SIGNAL: {signal}</b> 🚨---\n\n"
            f"<b>💰 Current Price:</b> <code>{current_price:{price_format}}</code>\n" f"<b>⏰ Timeframe:</b> {TIMEFRAME}\n"
            f"\n\n<b>🤖 ML PREDICTION</b>\n" f"<b>Forecast:</b> {ml_prediction}\n" f"<b>Confidence:</b> N/A\n"
            f"\n\n<b>📊 TECHNICAL &amp; KEY LEVELS</b>\n" f"{trend_emoji} <b>Trend (SMA 9/20):</b> {trend}\n" f"{proximity_msg}\n\n"
            f"{cpr_text}\n" f"----------------------------------------\n" f"<i>Exchange: {EXCHANGE_ID.upper()} Public | Disclaimer: Using crypto proxy data.</i>")

        message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'); message = message.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>'); message = message.replace('&lt;code&gt;', '<code>').replace('&lt;/code&gt;', '</code>'); message = message.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        
        asyncio.run(send_message_async(message, 'HTML'))
        
        global bot_stats
        bot_stats['total_analyses'] += 1; bot_stats['last_analysis'] = datetime.now().isoformat(); bot_stats['status'] = "operational"

    except Exception as e:
        traceback.print_exc()
        global bot_stats
        bot_stats['status'] = f"Fatal Error in Analysis: {str(e)[:40]}..."
        print(f"❌ Error generating signal for {symbol}: {e}")
        
        diagnostic_message = (f"❌ <b>FATAL PROXY ANALYSIS ERROR for {symbol}</b> ❌\n\n" f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S UTC')}\n" f"<b>Issue:</b> The calculation failed.\n\n" f"<b>Source Trace:</b>\n<code>{str(e)[:150]}</code>")
        asyncio.run(send_message_async(diagnostic_message, 'HTML'))


def start_scheduler_thread():
    """Starts the BackgroundScheduler in a separate thread."""
    
    # 1. Initialize ML model synchronously
    global ML_MODEL, SCALER

    print("\n⏳ Preparing and training Machine Learning Model...")
    try:
        # Use the first proxy symbol for training
        ohlcv_train = exchange.fetch_ohlcv(SYMBOLS[0].strip(), TIMEFRAME, limit=600)
        df_train = pd.DataFrame(ohlcv_train, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df_train['close'] = pd.to_numeric(df_train['close'])
        df_train['fast_sma'] = df_train['close'].rolling(window=9).mean(); df_train['slow_sma'] = df_train['close'].rolling(window=20).mean(); df_train = df_train.dropna()
        ML_MODEL, SCALER = train_prediction_model(df_train)
    except Exception as e:
        print(f"❌ ML Model Training Failed: {e}. Continuing without ML.")
        ML_MODEL = None; SCALER = None


    # 2. Start the Scheduler (BackgroundScheduler manages its own threads)
    scheduler = BackgroundScheduler() 
    
    for symbol in SYMBOLS:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[symbol]) 
    
    scheduler.start()
    print("🚀 Scheduler started successfully.")
    
    # Run initial analysis immediately
    generate_and_send_signal(SYMBOLS[0].strip()) 
    if len(SYMBOLS) > 1: generate_and_send_signal(SYMBOLS[1].strip())
    
    global bot_stats
    bot_stats['status'] = "operational"
    
    # Keep the thread alive
    while True:
        time.sleep(2)


# =========================================================================
# === SECTION 2: FLASK APP AND EXECUTABLE LOGIC (FINAL EXECUTABLE BLOCK) ===
# =========================================================================

# 1. FLASK WEB SERVER (App object is defined here)
app = Flask(__name__) 

@app.route('/')
def home():
    return render_template_string("<h1>Forex Proxy Bot Status Page</h1>" + 
                                 f"<p>Status: {bot_stats['status']}</p>" + 
                                 f"<p>Analyses: {bot_stats['total_analyses']}</p>" +
                                 f"<p>Exchange: {bot_stats['exchange']}</p>" +
                                 f"<p>Last Analysis: {bot_stats['last_analysis'] or 'N/A'}</p>")

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/status')
def status():
    return jsonify(bot_stats), 200


# 2. CRITICAL STARTUP CODE (Thread Start - The last lines of execution)
try:
    scheduler_thread = threading.Thread(target=start_scheduler_thread, daemon=True)
    scheduler_thread.start()
    print("✅ Gunicorn loading Flask app. Scheduler thread initialized.")
except Exception as e:
    print(f"FATAL THREAD START ERROR: {e}")
    bot_stats['status'] = f"FATAL ERROR: {str(e)[:40]}"
