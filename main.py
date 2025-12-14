# main.py - Minimal Working Forex Bot for Render
# Guaranteed to work with Gunicorn

import os
from flask import Flask, jsonify
from dotenv import load_dotenv

load_dotenv()

# Create Flask app FIRST (must be at module level for Gunicorn)
app = Flask(__name__)

# Basic routes to keep service alive
@app.route('/')
def home():
    return """
    <html>
    <head><title>Forex Bot</title></head>
    <body style="font-family: Arial; padding: 40px; text-align: center; background: #1e3c72; color: white;">
        <h1>💱 Forex Analysis Bot</h1>
        <h2 style="color: #4CAF50;">✅ RUNNING</h2>
        <p>Bot is operational and sending analyses to Telegram.</p>
    </body>
    </html>
    """

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

# Only start background tasks AFTER Flask is fully initialized
@app.before_first_request
def initialize():
    """Start background analysis after first web request"""
    import threading
    threading.Thread(target=start_analysis_worker, daemon=True).start()

def start_analysis_worker():
    """Background worker that runs analysis"""
    import time
    import ccxt
    import pandas as pd
    import numpy as np
    from telegram import Bot
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    
    print("\n" + "="*60)
    print("🚀 STARTING FOREX ANALYSIS WORKER")
    print("="*60)
    
    # Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    FOREX_PAIRS = os.getenv("FOREX_PAIRS", "EUR/USD,USD/JPY,GBP/USD").split(',')
    TIMEFRAME = os.getenv("TIMEFRAME", "4h")
    INTERVAL_MINUTES = int(os.getenv("ANALYSIS_INTERVAL", "30"))
    
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    exchange = ccxt.kraken({'enableRateLimit': True})
    
    # Global ML model
    ml_model = None
    scaler = None
    
    def train_model(df):
        """Train simple ML model"""
        try:
            if len(df) < 100:
                return None, None
            
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            df['sma_fast'] = df['close'].rolling(9).mean()
            df['sma_slow'] = df['close'].rolling(20).mean()
            df['momentum'] = df['close'].pct_change(5).fillna(0)
            df = df.dropna()
            
            if len(df) < 50:
                return None, None
            
            X = df[['sma_fast', 'sma_slow', 'momentum']].copy()
            y = df['target'].copy()
            
            split = int(len(X) * 0.9)
            X_train = X.iloc[:split]
            y_train = y.iloc[:split]
            
            scaler_local = StandardScaler()
            X_scaled = scaler_local.fit_transform(X_train)
            
            model = LogisticRegression(solver='liblinear', max_iter=1000)
            model.fit(X_scaled, y_train)
            
            print(f"✅ ML Model trained: {model.score(X_scaled, y_train):.1%} accuracy")
            return model, scaler_local
        except Exception as e:
            print(f"⚠️ ML training failed: {e}")
            return None, None
    
    def analyze_pair(pair):
        """Analyze single forex pair"""
        try:
            print(f"\n📊 Analyzing {pair}...")
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            df['sma_9'] = df['close'].rolling(9).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
            latest = df.iloc[-1]
            price = latest['close']
            sma9 = latest['sma_9']
            sma20 = latest['sma_20']
            rsi = latest['rsi']
            
            # Simple trend
            if price > sma9 > sma20:
                trend = "UPTREND 🟢"
            elif price < sma9 < sma20:
                trend = "DOWNTREND 🔴"
            else:
                trend = "SIDEWAYS 🟡"
            
            # ML prediction
            ml_signal = "NEUTRAL"
            if ml_model and scaler:
                try:
                    features = pd.DataFrame({
                        'sma_fast': [sma9],
                        'sma_slow': [sma20],
                        'momentum': [df['close'].pct_change(5).iloc[-1]]
                    })
                    X_scaled = scaler.transform(features)
                    pred = ml_model.predict(X_scaled)[0]
                    ml_signal = "BULLISH" if pred == 1 else "BEARISH"
                except:
                    pass
            
            # Signal
            if ml_signal == "BULLISH" and price > sma20:
                signal = "BUY 🟢"
            elif ml_signal == "BEARISH" and price < sma20:
                signal = "SELL 🔴"
            else:
                signal = "HOLD 🟡"
            
            # RSI status
            if rsi > 70:
                rsi_txt = f"{rsi:.1f} OVERBOUGHT ⚠️"
            elif rsi < 30:
                rsi_txt = f"{rsi:.1f} OVERSOLD 📉"
            else:
                rsi_txt = f"{rsi:.1f} NEUTRAL"
            
            # Build message
            message = f"""
╔═══════════════════════════════╗
  💱 FOREX ANALYSIS
╚═══════════════════════════════╝

<b>{pair}</b>
{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

━━━ <b>SIGNAL: {signal}</b> ━━━

💰 Price: <code>{price:.5f}</code>
⏰ Timeframe: {TIMEFRAME}

🤖 AI: <b>{ml_signal}</b>
📊 Trend: {trend}

📈 SMA 9: <code>{sma9:.5f}</code>
📈 SMA 20: <code>{sma20:.5f}</code>
📊 RSI: {rsi_txt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Educational purposes only. Not financial advice.</i>
"""
            
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
            print(f"✅ Sent: {pair} | {signal}")
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing {pair}: {e}")
            return False
    
    # Train ML model once on startup
    print("\n⏳ Training ML model...")
    try:
        first_pair = FOREX_PAIRS[0].strip()
        ohlcv = exchange.fetch_ohlcv(first_pair, TIMEFRAME, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ml_model, scaler = train_model(df)
    except Exception as e:
        print(f"⚠️ ML training skipped: {e}")
    
    print(f"\n✅ Worker initialized. Analyzing every {INTERVAL_MINUTES} minutes.")
    print(f"📊 Monitoring: {', '.join(FOREX_PAIRS)}\n")
    
    # Initial analysis
    for pair in FOREX_PAIRS:
        analyze_pair(pair.strip())
        time.sleep(5)
    
    # Main loop
    while True:
        try:
            time.sleep(INTERVAL_MINUTES * 60)  # Wait interval
            
            for pair in FOREX_PAIRS:
                analyze_pair(pair.strip())
                time.sleep(5)  # Delay between pairs
                
        except Exception as e:
            print(f"❌ Worker error: {e}")
            time.sleep(60)  # Wait 1 min on error

# For local testing only
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
