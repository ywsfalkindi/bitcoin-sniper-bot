import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import time
import os
import sys
import requests
from datetime import datetime
from flask import Flask
from threading import Thread

# ==========================================
# 🌍 1. إعدادات السيرفر (Render Keep-Alive)
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 BTC Sniper V7 is RUNNING! [Status: Active]"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()

# ==========================================
# ⚙️ 2. إعدادات البوت والاتصال
# ==========================================
API_KEY = os.environ.get("API_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOL = 'BTC/USDT'
LEVERAGE = 5
RISK_PER_TRADE = 0.02
CONFIDENCE_THRESHOLD = 0.65

def send_msg(text):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"⚠️ Telegram Error: {e}")

def get_exchange():
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  # مهم جداً للعقود الآجلة
            'adjustForTimeDifference': True
        }
    })
    exchange.set_sandbox_mode(True)
    return exchange

# ==========================================
# 🧠 3. هندسة الميزات
# ==========================================
def feature_engineering_v7(df):
    data = df.copy()
    data['Returns'] = np.log(data['close'] / data['close'].shift(1))
    data['Range'] = (data['high'] - data['low']) / data['open']
    data['Vol_1H'] = data['Returns'].rolling(24).std()
    data['Vol_4H_Proxy'] = data['Returns'].rolling(24).std()
    data['Vol_Ratio'] = data['Vol_1H'] / (data['Vol_4H_Proxy'] + 1e-9)
    data['Close_Loc'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-9)
    data['Volume_Flow'] = np.where(data['Close_Loc'] > 0.5, data['volume'], -data['volume'])
    data['CVD_Proxy'] = data['Volume_Flow'].rolling(12).sum()
    data['RSI'] = data.ta.rsi(length=14)
    data['MFI'] = data.ta.mfi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14']
    change = data['close'].diff(10).abs()
    volatility = data['close'].diff().abs().rolling(10).sum()
    data['Efficiency_Ratio'] = change / (volatility + 1e-9)
    data.dropna(inplace=True)
    return data

def get_market_data(exchange):
    # استخدام try-except هنا أيضاً لتجنب توقف البوت بسبب أخطاء الاتصال العابرة
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe='1h', limit=500)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        bars_4h = exchange.fetch_ohlcv(SYMBOL, timeframe='4h', limit=100)
        df_4h = pd.DataFrame(bars_4h, columns=['ts', 'o', 'h', 'l', 'close_4h', 'v'])
        
        try:
            fund = float(exchange.fetch_funding_rate(SYMBOL)['fundingRate'])
        except:
            fund = 0.0001
            
        data = df.copy()
        last_4h_close = df_4h.iloc[-1]['close_4h']
        
        data['Returns'] = np.log(data['close'] / data['close'].shift(1))
        data['Range'] = (data['high'] - data['low']) / data['open']
        data['Vol_1H'] = data['Returns'].rolling(24).std()
        data['Vol_4H_Proxy'] = data['Returns'].rolling(24).std()
        data['Vol_Ratio'] = data['Vol_1H'] / (data['Vol_4H_Proxy'] + 1e-9)
        data['Close_Loc'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-9)
        data['Volume_Flow'] = np.where(data['Close_Loc'] > 0.5, data['volume'], -data['volume'])
        data['CVD_Proxy'] = data['Volume_Flow'].rolling(12).sum()
        data['RSI'] = data.ta.rsi(length=14)
        data['MFI'] = data.ta.mfi(length=14)
        data['ADX'] = data.ta.adx(length=14)['ADX_14']
        change = data['close'].diff(10).abs()
        volatility = data['close'].diff().abs().rolling(10).sum()
        data['Efficiency_Ratio'] = change / (volatility + 1e-9)
        
        data['Funding_x_Vol'] = fund * data['Vol_1H']
        data['Trend_4H'] = 1 if data['close'].iloc[-1] > last_4h_close else 0
        data['fundingRate'] = fund
        
        return data.iloc[-1]
    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        return None

def check_open_positions(exchange):
    try:
        # جلب المراكز بطريقة آمنة
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if float(pos['contracts']) > 0:
                return True, float(pos['entryPrice']), float(pos['unrealizedPnl'])
        return False, 0, 0
    except Exception as e:
        print(f"⚠️ Error checking positions: {e}")
        return False, 0, 0

# ==========================================
# 🚀 4. المحرك الرئيسي
# ==========================================
def run_bot_logic():
    print("==========================================")
    print("💎 BTC SNIPER V7 (RENDER + TELEGRAM)")
    print("==========================================")
    send_msg("🚀 **Bot Started on Render!** Waiting for signals...")
    
    model_path = 'models/btc_v7_ensemble.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    model = joblib.load(model_path)
    
    try:
        exchange = get_exchange()
        # 🟢 الإصلاح: محاولة ضبط الرافعة وتجاهل الخطأ إن وجد
        try:
            exchange.set_leverage(LEVERAGE, SYMBOL)
            print(f"✅ Leverage set to {LEVERAGE}x")
        except Exception as e:
            print(f"⚠️ Warning: Cannot set leverage via API ({e}). Using account default.")

        print(f"✅ Connected to Testnet.")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    while True:
        try:
            print(f"\n⏰ Scan: {datetime.now().strftime('%H:%M:%S')}")
            
            # التحقق من الصفقات
            has_pos, entry, pnl = check_open_positions(exchange)
            if has_pos:
                print(f"⚠️ Position OPEN. PnL: ${pnl:.2f}")
                time.sleep(60)
                continue
            
            # تحليل السوق
            row = get_market_data(exchange)
            if row is None:
                time.sleep(60)
                continue

            features = [
                'RSI', 'MFI', 'ADX', 'Efficiency_Ratio', 
                'Vol_Ratio', 'CVD_Proxy', 'fundingRate', 
                'Funding_x_Vol', 'Trend_4H', 'Range'
            ]
            
            if row[features].isnull().any():
                print("⚠️ Not enough data.")
                time.sleep(60)
                continue

            # التوقع
            X_live = pd.DataFrame([row[features]])
            prob = model.predict_proba(X_live)[0][1]
            price = row['close']
            
            print(f"📊 Price: ${price:,.2f} | Confidence: {prob*100:.1f}%")
            
            if prob >= CONFIDENCE_THRESHOLD:
                print("🚀 SIGNAL DETECTED!")
                
                balance = exchange.fetch_balance()['USDT']['free']
                atr = row['ATRr_14'] if 'ATRr_14' in row else price * 0.015
                
                sl_price = price - (atr * 1.5)
                tp_price = price + (atr * 3.0)
                
                risk_amt = balance * RISK_PER_TRADE
                amount_btc = (risk_amt / (price - sl_price))
                if amount_btc < 0.002: amount_btc = 0.002
                
                # تنفيذ الأوامر
                exchange.create_market_buy_order(SYMBOL, amount_btc)
                exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', amount_btc, params={'stopPrice': sl_price})
                exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', amount_btc, params={'stopPrice': tp_price})
                
                # 📨 إرسال رسالة تليجرام
                msg = (
                    f"🔥 **AUTO TRADE EXECUTED** 🔥\n"
                    f"🟢 **LONG BTC/USDT**\n"
                    f"💵 Entry: ${price:,.2f}\n"
                    f"🛡️ SL: ${sl_price:,.2f}\n"
                    f"🎯 TP: ${tp_price:,.2f}\n"
                    f"🤖 Confidence: {prob*100:.1f}%"
                )
                send_msg(msg)
                print("✅ Trade Sent!")
                
            time.sleep(60)

        except Exception as e:
            print(f"❌ Error in loop: {e}")
            # send_msg(f"⚠️ **Bot Error:** {str(e)}") # يمكن تفعيلها إذا أردت تنبيهات للأخطاء
            time.sleep(60)

if __name__ == "__main__":
    keep_alive()
    run_bot_logic()