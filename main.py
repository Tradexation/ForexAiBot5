# main.py - PREMIER FOREX AI QUANT V2.5 (Render Optimized)

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Bot
from flask import Flask, jsonify, render_template_string
import threading
import time
import traceback 

# --- ML Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FOREX_PAIRS = [p.strip() for p in os.getenv("FOREX_PAIRS", "EUR/USD,GBP/USD,USD/JPY").split(',')]
TIMEFRAME_MAIN = "4h"  # Major Trend
TIMEFRAME_ENTRY = "1h" # Entry Precision

# Initialize Bot and Exchange (Defaulting to Kraken for India/Forex Stability)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kraken({
    'enableRateLimit': True, 
    'rateLimit': 2000,
    'params': {'timeout': 20000} # Explicit 20s timeout
})

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": FOREX_PAIRS,
    "uptime_start": datetime.now().isoformat(),
    "version": "V2.5 Forex Elite Quant"
}

# =========================================================================
# === ADVANCED QUANT LOGIC ===
# =========================================================================

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

def calculate_cpr_levels(df_daily):
    """Calculates Pivot Points for Institutional Target Setting."""
    if df_daily.empty or len(df_daily) < 2: return None
    prev_day = df_daily.iloc[-2]
    H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    PP = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = PP - BC + PP
    return {
        'PP': PP, 'TC': TC, 'BC': BC,
        'R1': 2*PP - L, 'S1': 2*PP - H,
        'R2': PP + (H - L), 'S2': PP - (H - L)
    }

def fetch_data_safe(symbol, timeframe):
    """Robust fetcher with retries and symbol ID normalization."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not exchange.markets: exchange.load_markets()
            market_id = exchange.market(symbol)['id']
            ohlcv = exchange.fetch_ohlcv(market_id, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['sma9'] = df['close'].rolling(9).mean()
            df['sma20'] = df['close'].rolling(20).mean()
            return df.dropna()
        except Exception:
            if attempt < max_retries - 1: time.sleep(5)
    return pd.DataFrame()

# =========================================================================
# === MULTI-TIMEFRAME CONFLUENCE ENGINE ===
# =========================================================================

def generate_and_send_signal(symbol):
    global bot_stats
    try:
        # 1. Multi-Timeframe Confluence
        df_4h = fetch_data_safe(symbol, TIMEFRAME_MAIN)
        df_1h = fetch_data_safe(symbol, TIMEFRAME_ENTRY)
        
        # 2. Daily Data for CPR Targets
        if not exchange.markets: exchange.load_markets()
        market_id = exchange.market(symbol)['id']
        ohlcv_d = exchange.fetch_ohlcv(market_id, '1d', limit=5)
        df_d = pd.DataFrame(ohlcv_d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        cpr = calculate_cpr_levels(df_d)

        if df_4h.empty or df_1h.empty or cpr is None: return

        # 3. Analyze Trend Confluence
        price = df_4h.iloc[-1]['close']
        trend_4h = "BULLISH" if df_4h.iloc[-1]['sma9'] > df_4h.iloc[-1]['sma20'] else "BEARISH"
        trend_1h = "BULLISH" if df_1h.iloc[-1]['sma9'] > df_1h.iloc[-1]['sma20'] else "BEARISH"
        
        # 4. Master Logic
        signal = "HOLD / WAIT"
        emoji = "⏳"
        
        if trend_4h == "BULLISH" and trend_1h == "BULLISH" and price > cpr['PP']:
            signal = "STRONG BUY"
            emoji = "🚀"
        elif trend_4h == "BEARISH" and trend_1h == "BEARISH" and price < cpr['PP']:
            signal = "STRONG SELL"
            emoji = "🔻"

        # 5. Risk Management Targets
        is_buy = "BUY" in signal
        tp1 = cpr['R1'] if is_buy else cpr['S1']
        tp2 = cpr['R2'] if is_buy else cpr['S2']
        sl = min(cpr['BC'], cpr['TC']) if is_buy else max(cpr['BC'], cpr['TC'])
        
        decimals = 5 if 'JPY' not in symbol else 3

        # --- PREMIUM SIGNAL TEMPLATE ---
        message = (
            f"╔════════════════════════════════╗\n"
            f"  🌍 <b>PREMIER FOREX AI QUANT</b>\n"
            f"╚════════════════════════════════╝\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Rate:</b> <code>{price:.{decimals}f}</code>\n\n"
            f"--- 🚨 {emoji} <b>SIGNAL: {signal}</b> 🚨 ---\n\n"
            f"<b>📈 CONFLUENCE ANALYSIS:</b>\n"
            f"• 4H Trend: <code>{trend_4h}</code>\n"
            f"• 1H Trend: <code>{trend_1h}</code>\n"
            f"• Pivot: {'Above' if price > cpr['PP'] else 'Below'} PP\n\n"
            f"<b>🎯 TARGET LEVELS:</b>\n"
            f"✅ <b>Take Profit 1:</b> <code>{tp1:.{decimals}f}</code>\n"
            f"🔥 <b>Take Profit 2:</b> <code>{tp2:.{decimals}f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{sl:.{decimals}f}</code>\n\n"
            f"----------------------------------------\n"
            f"<i>Verified AI Forex Analysis V2.5 Elite</i>"
        )

        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML'))
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

# =========================================================================
# === GUNICORN-SAFE INITIALIZATION ===
# =========================================================================

def start_bot():
    print(f"🚀 Initializing {bot_stats['version']}...")
    scheduler = BackgroundScheduler()
    for s in FOREX_PAIRS:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[s])
    scheduler.start()
    
    # Run immediate baseline check in separate threads
    for s in FOREX_PAIRS:
        threading.Thread(target=generate_and_send_signal, args=(s,)).start()

# Start bot outside main block for Gunicorn support
start_bot()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("""
        <body style="font-family:sans-serif; background:#020617; color:#f8fafc; text-align:center; padding-top:100px;">
            <div style="background:#0f172a; display:inline-block; padding:40px; border-radius:12px; border:1px solid #1e293b;">
                <h1 style="color:#38bdf8;">Forex AI Pro Dashboard</h1>
                <p>Status: <span style="color:#4ade80;">Active</span> | Version: {{v}}</p>
                <hr style="border-color:#1e293b;">
                <p>Analyses Streamed: <b>{{a}}</b></p>
                <p style="font-size:0.8em; color:#94a3b8;">{{t}}</p>
            </div>
        </body>
    """, a=bot_stats['total_analyses'], v=bot_stats['version'], t=bot_stats['last_analysis'])

@app.route('/health')
def health(): return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
