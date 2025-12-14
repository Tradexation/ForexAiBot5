# main.py - Gunicorn-Compatible Forex Analysis Bot
# Fixed for Render Web Service deployment

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
import threading
import traceback
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Bot
from flask import Flask, jsonify, render_template_string
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

# ========== CONFIGURATION ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FOREX_PAIRS = os.getenv("FOREX_PAIRS", "EUR/USD,USD/JPY,GBP/USD,USD/CHF,AUD/USD").split(',')
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
ANALYSIS_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", "30"))

# Initialize
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kraken({'enableRateLimit': True, 'rateLimit': 1000})

# Global variables
ML_MODEL = None
SCALER = None
scheduler = None
scheduler_started = False

print(f"✅ Configuration loaded. Monitoring {len(FOREX_PAIRS)} pairs.")

# ========== FLASK APP (Must be global for Gunicorn) ==========
app = Flask(__name__)

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_pairs": FOREX_PAIRS,
    "uptime_start": datetime.now().isoformat()
}

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Forex Bot</title>
    <style>
        body { font-family: Arial; background: linear-gradient(135deg, #1e3c72, #2a5298); 
               color: white; padding: 40px; text-align: center; }
        .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1);
                     padding: 40px; border-radius: 20px; }
        h1 { font-size: 2.5em; }
        .pair { display: inline-block; background: #4CAF50; padding: 10px 20px; 
                margin: 5px; border-radius: 20px; }
        .stat { font-size: 3em; color: #4CAF50; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>💱 Forex Analysis Bot</h1>
        <p>Professional Market Intelligence</p>
        <div class="stat">{{ stats.total_analyses }}</div>
        <p>Total Analyses</p>
        <h3>Monitoring:</h3>
        {% for pair in pairs %}
        <span class="pair">{{ pair }}</span>
        {% endfor %}
        <p style="margin-top: 30px;">Status: <strong>{{ stats.status|upper }}</strong></p>
        <p>Last: {{ stats.last_analysis or 'Not yet' }}</p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_DASHBOARD, stats=bot_stats, pairs=FOREX_PAIRS)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/status')
def status():
    return jsonify(bot_stats), 200

# ========== ML TRAINING ==========

def train_ml_model(df):
    """Train ML model"""
    global SCALER
    
    try:
        if len(df) < 100:
            print(f"⚠️ Not enough data for ML training ({len(df)} rows)")
            return None, None

        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        df['sma_cross'] = np.where(df['fast_sma'] > df['slow_sma'], 1, 0)
        df['price_position'] = np.where(df['close'] > df['fast_sma'], 1, 0)
        
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(20, min_periods=1).std().fillna(0)
        df['volatility'] = df['volatility'].replace([np.inf, -np.inf], 0)
        df['volatility'] = np.clip(df['volatility'], 0, 0.1)
        
        df['momentum'] = df['close'].pct_change(5).fillna(0)
        df['momentum'] = np.clip(df['momentum'], -0.05, 0.05)
        
        df = df.dropna()
        
        if len(df) < 50:
            return None, None
        
        X = df[['sma_cross', 'price_position', 'volatility', 'momentum']].copy()
        y = df['target'].copy()
        
        split_idx = int(len(X) * 0.9)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        SCALER = StandardScaler()
        X_train_scaled = SCALER.fit_transform(X_train)
        
        model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        print(f"✅ ML Model trained: {model.score(X_train_scaled, y_train):.2%} accuracy")
        return model, SCALER
        
    except Exception as e:
        print(f"❌ ML training error: {e}")
        return None, None

# ========== TECHNICAL ANALYSIS ==========

def calculate_pivot_points(df_daily):
    """Calculate pivot points"""
    if df_daily.empty or len(df_daily) < 2:
        return None
    
    prev = df_daily.iloc[-2]
    H, L, C = prev['high'], prev['low'], prev['close']
    PP = (H + L + C) / 3.0
    
    return {
        'PP': PP,
        'R1': (2 * PP) - L,
        'S1': (2 * PP) - H,
        'R2': PP + (H - L),
        'S2': PP - (H - L)
    }

def calculate_atr(df, period=14):
    """Calculate ATR"""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if len(atr) > 0 else 0
    except:
        return 0

def fetch_and_analyze(pair):
    """Fetch data and analyze"""
    try:
        # Fetch main data
        ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        df['fast_sma'] = df['close'].rolling(9).mean()
        df['slow_sma'] = df['close'].rolling(20).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        
        if len(df) < 20:
            return None
        
        # Daily data for pivots
        ohlcv_daily = exchange.fetch_ohlcv(pair, '1d', limit=20)
        df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        pivots = calculate_pivot_points(df_daily)
        atr = calculate_atr(df)
        
        # Analysis
        latest = df.iloc[-1]
        current_price = latest['close']
        fast_sma = latest['fast_sma']
        slow_sma = latest['slow_sma']
        rsi = latest['rsi']
        
        # Trend
        if fast_sma > slow_sma and current_price > fast_sma:
            trend = "STRONG UPTREND"
            trend_emoji = "🟢🟢"
        elif fast_sma > slow_sma:
            trend = "UPTREND"
            trend_emoji = "🟢"
        elif fast_sma < slow_sma and current_price < fast_sma:
            trend = "STRONG DOWNTREND"
            trend_emoji = "🔴🔴"
        elif fast_sma < slow_sma:
            trend = "DOWNTREND"
            trend_emoji = "🔴"
        else:
            trend = "SIDEWAYS"
            trend_emoji = "🟡"
        
        # ML Prediction
        ml_pred = "NEUTRAL"
        ml_conf = "N/A"
        
        if ML_MODEL and SCALER:
            try:
                returns = df['close'].iloc[-20:].pct_change().dropna()
                volatility = float(returns.std()) if len(returns) > 0 else 0.0
                volatility = np.clip(volatility, 0, 0.1)
                
                momentum = df['close'].pct_change(5).iloc[-1]
                momentum = np.clip(momentum, -0.05, 0.05)
                
                features = pd.DataFrame({
                    'sma_cross': [1 if fast_sma > slow_sma else 0],
                    'price_position': [1 if current_price > fast_sma else 0],
                    'volatility': [volatility],
                    'momentum': [momentum]
                })
                
                X_scaled = SCALER.transform(features)
                prediction = ML_MODEL.predict(X_scaled)[0]
                prob = ML_MODEL.predict_proba(X_scaled)[0]
                
                if prediction == 1:
                    ml_pred = "BULLISH"
                    ml_conf = f"{prob[1]*100:.0f}%"
                else:
                    ml_pred = "BEARISH"
                    ml_conf = f"{prob[0]*100:.0f}%"
            except Exception as e:
                print(f"ML prediction error: {e}")
        
        # Signal
        signal = "HOLD"
        signal_emoji = "🟡"
        pp = pivots['PP'] if pivots else current_price
        
        if "BULLISH" in ml_pred and current_price > pp and rsi < 70:
            signal = "BUY"
            signal_emoji = "🟢"
        elif "BEARISH" in ml_pred and current_price < pp and rsi > 30:
            signal = "SELL"
            signal_emoji = "🔴"
        
        # RSI status
        if rsi > 70:
            rsi_status = "OVERBOUGHT ⚠️"
        elif rsi < 30:
            rsi_status = "OVERSOLD 📉"
        else:
            rsi_status = "NEUTRAL ➡️"
        
        # Pip calculations
        pip_value = 0.0001 if 'JPY' not in pair else 0.01
        pips_to_r1 = (pivots['R1'] - current_price) / pip_value if pivots else 0
        pips_to_s1 = (current_price - pivots['S1']) / pip_value if pivots else 0
        
        return {
            'pair': pair,
            'price': current_price,
            'trend': trend,
            'trend_emoji': trend_emoji,
            'signal': signal,
            'signal_emoji': signal_emoji,
            'ml_pred': ml_pred,
            'ml_conf': ml_conf,
            'rsi': rsi,
            'rsi_status': rsi_status,
            'atr': atr,
            'fast_sma': fast_sma,
            'slow_sma': slow_sma,
            'pivots': pivots,
            'pips_to_r1': pips_to_r1,
            'pips_to_s1': pips_to_s1
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {pair}: {e}")
        traceback.print_exc()
        return None

def send_analysis_sync(pair):
    """Synchronous function for scheduler"""
    try:
        print(f"\n{'='*60}")
        print(f"📊 Analyzing {pair}...")
        
        analysis = fetch_and_analyze(pair)
        
        if not analysis:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"⚠️ Failed to analyze {pair}")
            return
        
        # Build message
        pivots = analysis['pivots']
        message = (
            f"╔═══════════════════════════════╗\n"
            f"  💱 <b>FOREX MARKET ANALYSIS</b>\n"
            f"╚═══════════════════════════════╝\n\n"
            f"<b>{analysis['pair']}</b> | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"━━━ {analysis['signal_emoji']} <b>SIGNAL: {analysis['signal']}</b> {analysis['signal_emoji']} ━━━\n\n"
            f"<b>💰 Price:</b> <code>{analysis['price']:.5f}</code>\n"
            f"<b>⏰ Timeframe:</b> {TIMEFRAME}\n"
            f"<b>📏 ATR:</b> {analysis['atr']:.5f}\n\n"
            f"<b>🤖 AI PREDICTION</b>\n"
            f"  • Forecast: <b>{analysis['ml_pred']}</b>\n"
            f"  • Confidence: {analysis['ml_conf']}\n\n"
            f"<b>📊 TECHNICAL</b>\n"
            f"  {analysis['trend_emoji']} Trend: <b>{analysis['trend']}</b>\n"
            f"  • SMA 9: <code>{analysis['fast_sma']:.5f}</code>\n"
            f"  • SMA 20: <code>{analysis['slow_sma']:.5f}</code>\n"
            f"  • RSI: <code>{analysis['rsi']:.1f}</code> - {analysis['rsi_status']}\n\n"
            f"<b>🎯 KEY LEVELS</b>\n"
            f"  • R1: <code>{pivots['R1']:.5f}</code> (+{analysis['pips_to_r1']:.0f} pips)\n"
            f"  • <b>PP: <code>{pivots['PP']:.5f}</code></b>\n"
            f"  • S1: <code>{pivots['S1']:.5f}</code> (-{analysis['pips_to_s1']:.0f} pips)\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>⚠️ Educational purposes only. Not financial advice.</i>"
        )
        
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"
        
        print(f"✅ Analysis sent: {pair} | {analysis['signal']} | {analysis['ml_pred']}")
        
    except Exception as e:
        print(f"❌ Send error: {e}")
        traceback.print_exc()

def initialize_ml():
    """Initialize ML model on startup"""
    global ML_MODEL, SCALER
    
    print("\n⏳ Training ML model...")
    try:
        first_pair = FOREX_PAIRS[0].strip()
        ohlcv = exchange.fetch_ohlcv(first_pair, TIMEFRAME, limit=600)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['fast_sma'] = df['close'].rolling(9).mean()
        df['slow_sma'] = df['close'].rolling(20).mean()
        df = df.dropna()
        
        ML_MODEL, SCALER = train_ml_model(df)
    except Exception as e:
        print(f"⚠️ ML training failed: {e}")
        ML_MODEL = None
        SCALER = None

def start_scheduler():
    """Start the background scheduler"""
    global scheduler, scheduler_started
    
    if scheduler_started:
        return
    
    print("\n🚀 Starting scheduler...")
    
    # Initialize ML first
    initialize_ml()
    
    # Create scheduler
    scheduler = BackgroundScheduler()
    
    # Add jobs for each pair
    for pair in [p.strip() for p in FOREX_PAIRS]:
        scheduler.add_job(
            send_analysis_sync,
            'cron',
            minute=f'*/{ANALYSIS_INTERVAL}',
            args=[pair],
            id=f'analyze_{pair.replace("/", "_")}'
        )
    
    scheduler.start()
    scheduler_started = True
    
    print(f"✅ Scheduler started. Analysis every {ANALYSIS_INTERVAL} minutes.")
    print(f"📊 Monitoring: {', '.join(FOREX_PAIRS)}")
    
    # Run initial analysis
    for pair in [p.strip() for p in FOREX_PAIRS]:
        send_analysis_sync(pair)

# ========== STARTUP HOOK ==========

@app.before_first_request
def on_startup():
    """Start scheduler when first request is received"""
    if not scheduler_started:
        threading.Thread(target=start_scheduler, daemon=True).start()

# ========== FOR GUNICORN ==========

if __name__ == "__main__":
    # This runs for local testing only
    port = int(os.environ.get('PORT', 10000))
    start_scheduler()
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # This runs when imported by Gunicorn
    print("✅ Flask app initialized for Gunicorn")
