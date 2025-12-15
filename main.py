# main.py - Forex Edition: Based on the WORKING Crypto Structure

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

# --- FOREX CONFIGURATION ---
# CRITICAL FIX: Use Forex pairs and a stable Forex exchange ID (FXCM/OANDA alternative)
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "fxcm") 
SYMBOLS = os.getenv("FOREX_PAIRS", "EUR/USD,USD/JPY,GBP/USD,AUD/USD,USD/CAD").split(',') 
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
DAILY_TIMEFRAME = '1d'
ANALYSIS_INTERVAL = 30 

# Initialize Bot and Exchange (Uses FXCM or the ID set in .env)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

exchange_config = {
    'enableRateLimit': True,
    'rateLimit': 1000, 
    'apiKey': os.getenv("FX_API_KEY"),
    'secret': os.getenv("FX_SECRET"),
    # Add other necessary config if required by the broker
}

# Dynamically instantiate the exchange
try:
    exchange = getattr(ccxt, EXCHANGE_ID)(exchange_config)
    exchange.load_markets() 
    print(f"✅ {EXCHANGE_ID.upper()} markets loaded successfully.")
except Exception as e:
    print(f"❌ Failed to initialize/load markets for {EXCHANGE_ID}: {e}. Check .env credentials.")
    exit(1) # Crash if exchange fails to prevent the NameError later

# Global ML Model and Scaler (must be global for prediction)
ML_MODEL = None
SCALER = None

# ========== FLASK WEB SERVER & STATUS TRACKING ==========
app = Flask(__name__) 

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": SYMBOLS,
    "uptime_start": datetime.now().isoformat(),
    "exchange": EXCHANGE_ID.upper()
}

@app.route('/')
def home():
    return render_template_string("<h1>Forex Bot Status Page</h1>" + 
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

# ========== ML TRAINING FUNCTION (PLACED BEFORE CALLS) ==========

def train_prediction_model(df):
    """Trains a Logistic Regression model and returns the model and scaler."""
    global SCALER
    
    try:
        if len(df) < 100:
            print(f"⚠️ Not enough data ({len(df)} rows, need 100+) for ML training. Skipping.")
            return None, None

        # 1. Target Definition (y)
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        # 2. Feature Engineering (X)
        df['fast_over_slow'] = np.where(df['fast_sma'] > df['slow_sma'], 1, 0)
        df['close_over_fast'] = np.where(df['close'] > df['fast_sma'], 1, 0)
        
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(20, min_periods=1).std().fillna(0)
        df['volatility'] = df['volatility'].replace([np.inf, -np.inf], 0)
        df['volatility'] = np.clip(df['volatility'], 0, 1) 
        df = df.dropna()
        
        if len(df) < 50:
            print(f"⚠️ After cleaning, only {len(df)} rows left. Need 50+. Skipping ML.")
            return None, None
        
        X = df[['fast_over_slow', 'close_over_fast', 'volatility']].copy()
        y = df['target'].copy()
        
        split_idx = int(len(X) * 0.9)
        X_train = X.iloc[:split_idx]; y_train = y.iloc[:split_idx]

        # 3. Scaling
        SCALER = StandardScaler()
        X_train_scaled = SCALER.fit_transform(X_train)

        # 4. Training
        model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_train_scaled, y_train)
        print(f"✅ ML Model trained successfully. Training Accuracy: {accuracy:.2%}")
        return model, SCALER
        
    except Exception as e:
        print(f"❌ ML Training Error: {e}")
        traceback.print_exc()
        return None, None

# 1. CPR Calculation Function
def calculate_cpr_levels(df_daily):
    """Calculates Daily Pivot Points (PP, TC, BC, R/S levels) from previous day's data."""
    if df_daily.empty or len(df_daily) < 2: return None

    prev_day = df_daily.iloc[-2]; H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    PP = (H + L + C) / 3.0; BC = (H + L) / 2.0; TC = PP - BC + PP
    R1 = 2 * PP - L; S1 = 2 * PP - H; R2 = PP + (H - L); S2 = PP - (H - L); R3 = H + 2 * (PP - L); S3 = L - 2 * (H - PP)
    
    return {'PP': PP, 'TC': TC, 'BC': BC, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}

# 2. Data Fetching and Preparation Function
def fetch_and_prepare_data(symbol, timeframe, daily_timeframe='1d', limit=500):
    """Fetches main chart data, prepares for analysis, and calculates SMAs."""
    
    # CRITICAL: Use normalized symbol ID
    try:
        fx_symbol_id = exchange.markets[symbol]['id']
    except KeyError:
        print(f"Error: Symbol {symbol} not found in exchange market list.")
        return pd.DataFrame(), None

    ohlcv = exchange.fetch_ohlcv(fx_symbol_id, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True); df = df.dropna()
    df['fast_sma'] = df['close'].rolling(window=9).mean(); df['slow_sma'] = df['close'].rolling(window=20).mean()
    df = df.dropna(); 
    
    if len(df) < 20: return pd.DataFrame(), None
    
    ohlcv_daily = exchange.fetch_ohlcv(fx_symbol_id, daily_timeframe, limit=20) 
    df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df_daily.set_index('timestamp', inplace=True)
    cpr_levels = calculate_cpr_levels(df_daily)
    return df, cpr_levels

# 3. Trend and Signal Generation
def get_trend_and_signal(df, cpr_levels):
    """Determines trend via SMA crossover and incorporates ML prediction."""
    
    latest = df.iloc[-1]; current_price = latest['close']; fast_sma = latest['fast_sma']; slow_sma = latest['slow_sma']
    ml_prediction = "NEUTRAL (No Model)"; ml_confidence = "N/A"
    
    if ML_MODEL is not None and SCALER is not None:
        try:
            # 1. Calculate volatility safely
            close_prices_recent = df['close'].iloc[-20:]
            if len(close_prices_recent) < 20: current_volatility = 0.0
            else:
                 returns = close_prices_recent.pct_change().dropna(); current_volatility = float(returns.std())
                 if np.isnan(current_volatility) or np.isinf(current_volatility): current_volatility = 0.0
                 current_volatility = np.clip(current_volatility, 0, 1)

            # 2. Build features
            is_fast_over_slow = 1 if fast_sma > slow_sma else 0; is_close_over_fast = 1 if current_price > fast_sma else 0
            latest_features = pd.DataFrame({'fast_over_slow': [is_fast_over_slow], 'close_over_fast': [is_close_over_fast], 'volatility': [current_volatility]})
            
            # 3. Scaling and Prediction
            X_predict_scaled = SCALER.transform(latest_features)
            prediction = ML_MODEL.predict(X_predict_scaled)[0]; probability = ML_MODEL.predict_proba(X_predict_scaled)[0]
            bullish_prob = probability[1]; bearish_prob = probability[0]
            
            # 4. Final Prediction Output
            if prediction == 1 and bullish_prob > 0.55: ml_prediction = "BULLISH"; ml_confidence = f"{bullish_prob*100:.0f}%"
            elif prediction == 0 and bearish_prob > 0.55: ml_prediction = "BEARISH"; ml_confidence = f"{bearish_prob*100:.0f}%"
            else: ml_prediction = "NEUTRAL"; ml_confidence = f"{max(bullish_prob, bearish_prob)*100:.0f}%"
        except Exception as e:
            print(f"❌ ML PREDICTION FAILED: {e}"); traceback.print_exc(); ml_prediction = "NEUTRAL (Error)"; ml_confidence = "Error"
            
    # --- Trend Assessment ---
    trend = "Neutral"; trend_emoji = "🟡"
    if fast_sma > slow_sma: trend = "Uptrend"; trend_emoji = "🟢"
    elif fast_sma < slow_sma: trend = "Downtrend"; trend_emoji = "🔴"

    # --- Final Signal Generation ---
    pp = cpr_levels.get('PP', 'N/A')
    proximity_msg = ""; price_format = ".5f" # Forex format
    if pp != 'N/A':
        distance_to_pp = current_price - pp
        if abs(distance_to_pp / pp) < 0.0005: proximity_msg = "Price is near the Central Pivot Point (PP)."
        elif distance_to_pp > 0: proximity_msg = f"Price is Above PP ({pp:{price_format}})."
        else: proximity_msg = f"Price is Below PP ({pp:{price_format}})."
            
    signal = "HOLD"; signal_emoji = "🟡"
    if "BULLISH" in ml_prediction and current_price > pp: signal = "STRONG BUY"; signal_emoji = "🚀"
    elif "BEARISH" in ml_prediction and current_price < pp: signal = "STRONG SELL"; signal_emoji = "🔻"
        
    return trend, trend_emoji, proximity_msg, signal, signal_emoji, ml_prediction, ml_confidence

# 4. ASYNC SCHEDULER FUNCTIONS

async def generate_and_send_signal(symbol):
    """Fetches data, runs analysis, and sends the Telegram message."""
    
    try:
        print(f"\n{'='*60}"); print(f"🔍 Analyzing {symbol}...")
        df, cpr_levels = fetch_and_prepare_data(symbol, TIMEFRAME)
        
        if df.empty or cpr_levels is None:
            message = f"🚨 Data Fetch/Processing Error for {symbol}. Could not generate signal (Insufficient clean data)."
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            return

        trend, trend_emoji, proximity_msg, signal, signal_emoji, ml_prediction, ml_confidence = get_trend_and_signal(df, cpr_levels)
        current_price = df.iloc[-1]['close']; price_format = ".5f" if current_price < 10 else ".4f"
        
        cpr_text = (f"<b>Daily CPR Levels:</b>\n" f"  - <b>PP (Pivot Point):</b> <code>{cpr_levels['PP']:{price_format}}</code>\n" f"  - <b>R1/S1:</b> <code>{cpr_levels['R1']:{price_format}}</code> / <code>{cpr_levels['S1']:{price_format}}</code>\n" f"  - <b>R2/S2:</b> <code>{cpr_levels['R2']:{price_format}}</code> / <code>{cpr_levels['S2']:{price_format}}</code>\n")
        message = (
            f"╔════════════════════════════════╗\n" f"  🧠 <b>FOREX AI INTELLIGENCE REPORT</b>\n" f"╚════════════════════════════════╝\n\n"
            f"<b>{symbol}</b> | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"---🚨 <b>{signal_emoji} FINAL SIGNAL: {signal}</b> 🚨---\n\n"
            f"<b>💰 Current Price:</b> <code>{current_price:{price_format}}</code>\n" f"<b>⏰ Timeframe:</b> {TIMEFRAME}\n"
            f"\n<b>🤖 ML PREDICTION</b>\n" f"<b>Forecast:</b> {ml_prediction}\n" f"<b>Confidence:</b> {ml_confidence}\n"
            f"\n<b>📊 TECHNICAL &amp; KEY LEVELS</b>\n" f"{trend_emoji} <b>Trend (SMA 9/20):</b> {trend}\n" f"{proximity_msg}\n\n"
            f"{cpr_text}\n" f"----------------------------------------\n" f"<i>Exchange: {EXCHANGE_ID.upper()} | Disclaimer: This analysis is for educational purposes only.</i>")

        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"
        
        print(f"✅ Analysis sent: {symbol} | Signal: {signal} | ML: {ml_prediction} ({ml_confidence})")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Error generating signal for {symbol}: {e}"); print(error_trace)
        
        diagnostic_message = (f"❌ <b>FATAL FOREX ANALYSIS ERROR for {symbol}</b> ❌\n\n" f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S UTC')}\n" f"<b>Issue:</b> The calculation thread crashed.\n\n" f"<b>Error:</b> <code>{str(e)[:200]}</code>")
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=diagnostic_message, parse_mode='HTML')


async def start_scheduler_loop():
    """Sets up the scheduler and keeps the asyncio loop running."""
    
    global ML_MODEL; global SCALER

    print("\n⏳ Preparing and training Machine Learning Model...")
    try:
        # Fetch data for training (using the first symbol)
        ohlcv_train = exchange.fetch_ohlcv(SYMBOLS[0].strip(), TIMEFRAME, limit=600)
        df_train = pd.DataFrame(ohlcv_train, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df_train['close'] = pd.to_numeric(df_train['close'])
        df_train['fast_sma'] = df_train['close'].rolling(window=9).mean(); df_train['slow_sma'] = df_train['close'].rolling(window=20).mean(); df_train = df_train.dropna()
        ML_MODEL, SCALER = train_prediction_model(df_train)
        if ML_MODEL is None: print("⚠️ Bot will run WITHOUT ML predictions (using technical analysis only)")
    except Exception as e:
        print(f"❌ ML Model Training Failed: {e}"); traceback.print_exc(); ML_MODEL = None; SCALER = None

    # --- Start the scheduler loop ---
    scheduler = AsyncIOScheduler()
    for symbol in SYMBOLS:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[symbol]) 
    
    scheduler.start()
    print("🚀 Scheduler started successfully. Signals every 30 minutes.")

    # Run initial analysis immediately
    for symbol in SYMBOLS:
        await generate_and_send_signal(symbol)
        await asyncio.sleep(5)

    # Keep the main thread running
    while True:
        await asyncio.sleep(60)


# 5. CRITICAL STARTUP THREAD
def start_asyncio_thread():
    """Target function for the background thread."""
    try:
        asyncio.run(start_scheduler_loop())
    except Exception as e:
        print(f"FATAL SCHEDULER ERROR: {e}"); traceback.print_exc()

# This thread starts immediately when Gunicorn loads the 'app' instance
scheduler_thread = threading.Thread(target=start_asyncio_thread, daemon=True)
scheduler_thread.start()

print("✅ Gunicorn loading Flask app. Scheduler thread initialized.")
