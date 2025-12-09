import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import time
import os
import sys
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
    # Render يعطي المنفذ تلقائياً، نستخدم 5000 كاحتياط
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    """تشغيل السيرفر في خيط منفصل حتى لا يوقف البوت"""
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()

# ==========================================
# ⚙️ 2. إعدادات البوت والاتصال
# ==========================================
# جلب المفاتيح من متغيرات البيئة في Render (للأمان)
# إذا كنت تجربه على جهازك، استبدل النص داخل الأقواس بمفاتيحك مباشرة
API_KEY = os.environ.get("API_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")

SYMBOL = 'BTC/USDT'
LEVERAGE = 5            # الرافعة المالية (تجريبية)
RISK_PER_TRADE = 0.02   # المخاطرة 2%
CONFIDENCE_THRESHOLD = 0.65 # العتبة التي حددناها في المعايرة

def get_exchange():
    """الاتصال بـ Binance Testnet"""
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'} # العقود الآجلة
    })
    exchange.set_sandbox_mode(True) # 👈 تفعيل الوضع التجريبي
    return exchange

# ==========================================
# 🧠 3. هندسة الميزات (يجب أن تطابق التدريب 100%)
# ==========================================
def feature_engineering_v7(df):
    data = df.copy()
    
    # المؤشرات الأساسية
    data['Returns'] = np.log(data['close'] / data['close'].shift(1))
    data['Range'] = (data['high'] - data['low']) / data['open']
    
    # Volatility Surface
    data['Vol_1H'] = data['Returns'].rolling(24).std()
    data['Vol_4H_Proxy'] = data['Returns'].rolling(24).std()
    data['Vol_Ratio'] = data['Vol_1H'] / (data['Vol_4H_Proxy'] + 1e-9)
    
    # Order Flow Proxy
    data['Close_Loc'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-9)
    data['Volume_Flow'] = np.where(data['Close_Loc'] > 0.5, data['volume'], -data['volume'])
    data['CVD_Proxy'] = data['Volume_Flow'].rolling(12).sum()
    
    # Momentum
    data['RSI'] = data.ta.rsi(length=14)
    data['MFI'] = data.ta.mfi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14']
    
    # Efficiency Ratio
    change = data['close'].diff(10).abs()
    volatility = data['close'].diff().abs().rolling(10).sum()
    data['Efficiency_Ratio'] = change / (volatility + 1e-9)
    
    # سيتم حساب funding_x_vol و trend_4h لاحقاً عند دمج البيانات الحية
    
    data.dropna(inplace=True)
    return data

def get_market_data(exchange):
    """جلب وتحضير البيانات الحية"""
    # جلب 1H
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe='1h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # جلب 4H
    bars_4h = exchange.fetch_ohlcv(SYMBOL, timeframe='4h', limit=100)
    df_4h = pd.DataFrame(bars_4h, columns=['ts', 'o', 'h', 'l', 'close_4h', 'v'])
    
    # جلب التمويل
    try:
        fund = float(exchange.fetch_funding_rate(SYMBOL)['fundingRate'])
    except:
        fund = 0.0001
        
    # تطبيق الهندسة
    data = df.copy()
    last_4h_close = df_4h.iloc[-1]['close_4h']
    
    # إعادة الحسابات للتأكد من التطابق
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
    
    # الميزات السياقية
    data['Funding_x_Vol'] = fund * data['Vol_1H']
    data['Trend_4H'] = 1 if data['close'].iloc[-1] > last_4h_close else 0
    data['fundingRate'] = fund
    
    return data.iloc[-1]

def check_open_positions(exchange):
    """هل توجد صفقة مفتوحة؟"""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if float(pos['contracts']) > 0:
                return True, float(pos['entryPrice']), float(pos['unrealizedPnl'])
        return False, 0, 0
    except Exception as e:
        print(f"⚠️ Error checking positions: {e}")
        return False, 0, 0

# ==========================================
# 🚀 4. المحرك الرئيسي (Main Loop)
# ==========================================
def run_bot_logic():
    print("==========================================")
    print("💎 BTC SNIPER V7 (RENDER + TESTNET)")
    print("==========================================")
    
    # التأكد من وجود النموذج
    model_path = 'models/btc_v7_ensemble.pkl'
    if not os.path.exists(model_path):
        print(f"❌ FATAL ERROR: Model not found at {model_path}")
        print("Please upload the 'models' folder to Render.")
        return

    model = joblib.load(model_path)
    
    try:
        exchange = get_exchange()
        exchange.set_leverage(LEVERAGE, SYMBOL)
        print(f"✅ Connected to Testnet. Leverage set to {LEVERAGE}x")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    while True:
        try:
            print(f"\n⏰ Scan: {datetime.now().strftime('%H:%M:%S')}")
            
            # 1. التحقق من الصفقات المفتوحة
            has_pos, entry, pnl = check_open_positions(exchange)
            if has_pos:
                print(f"⚠️ Position OPEN. Entry: ${entry:.2f} | PnL: ${pnl:.2f}")
                print("⏳ Waiting for TP/SL trigger...")
                time.sleep(60) # فحص كل دقيقة
                continue
            
            # 2. تحليل السوق
            row = get_market_data(exchange)
            
            features = [
                'RSI', 'MFI', 'ADX', 'Efficiency_Ratio', 
                'Vol_Ratio', 'CVD_Proxy', 'fundingRate', 
                'Funding_x_Vol', 'Trend_4H', 'Range'
            ]
            
            # التأكد من البيانات
            if row[features].isnull().any():
                print("⚠️ Not enough data (NaN detected). Waiting...")
                time.sleep(60)
                continue

            # 3. التوقع
            X_live = pd.DataFrame([row[features]])
            prob = model.predict_proba(X_live)[0][1]
            price = row['close']
            
            print(f"📊 Price: ${price:,.2f} | 🤖 AI Confidence: {prob*100:.2f}%")
            
            # 4. اتخاذ القرار
            if prob >= CONFIDENCE_THRESHOLD:
                print("🚀 SIGNAL DETECTED! Executing trade...")
                
                # جلب الرصيد
                balance = exchange.fetch_balance()['USDT']['free']
                atr = row['ATRr_14'] if 'ATRr_14' in row else price * 0.015
                
                # حساب الأهداف
                sl_price = price - (atr * 1.5)
                tp_price = price + (atr * 3.0)
                
                # حساب الكمية (Risk Management)
                risk_amt = balance * RISK_PER_TRADE
                sl_dist = price - sl_price
                amount_btc = (risk_amt / sl_dist)
                
                # الحد الأدنى للكمية (Binance Limit)
                if amount_btc < 0.002: amount_btc = 0.002
                
                # -------------------------
                # تنفيذ الأوامر (Atomic Execution)
                # -------------------------
                # 1. فتح الصفقة (Market Buy)
                print(f"🛒 Buying {amount_btc:.4f} BTC...")
                order = exchange.create_market_buy_order(SYMBOL, amount_btc)
                
                # 2. وضع وقف الخسارة
                print(f"🛡️ Setting SL at ${sl_price:.2f}...")
                exchange.create_order(
                    symbol=SYMBOL,
                    type='STOP_MARKET',
                    side='sell',
                    amount=amount_btc,
                    params={'stopPrice': sl_price}
                )
                
                # 3. وضع جني الأرباح
                print(f"🎯 Setting TP at ${tp_price:.2f}...")
                exchange.create_order(
                    symbol=SYMBOL,
                    type='TAKE_PROFIT_MARKET',
                    side='sell',
                    amount=amount_btc,
                    params={'stopPrice': tp_price}
                )
                
                print("✅ Trade Executed Successfully!")
                
            else:
                print(f"💤 No trade. Threshold is {CONFIDENCE_THRESHOLD}")
            
            # الانتظار دقيقة واحدة (Render يحتاج لنشاط مستمر)
            # UptimeRobot سيبقي السيرفر حياً، لكن الفحص المتكرر جيد
            time.sleep(60)

        except Exception as e:
            print(f"❌ Error in loop: {e}")
            time.sleep(60) # انتظار عند حدوث خطأ لتجنب الحظر

if __name__ == "__main__":
    # 1. تشغيل السيرفر الوهمي (Keep-Alive)
    keep_alive()
    
    # 2. تشغيل البوت (Main Logic)
    run_bot_logic()