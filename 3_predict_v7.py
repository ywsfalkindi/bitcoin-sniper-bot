import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import requests
import os
import time
from dotenv import load_dotenv

# تحميل إعدادات البيئة
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --- إعدادات القناص ---
CONFIDENCE_THRESHOLD = 0.65  # 👈 تم الضبط بناءً على المعايرة
CAPITAL = 1000               # رصيد المحفظة الافتراضي
RISK_PER_TRADE = 0.02        # المخاطرة 2% لكل صفقة

def send_msg(text):
    if not TELEGRAM_TOKEN: 
        print(f"\n📨 [Telegram Mock]: {text}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"⚠️ Telegram Error: {e}")

def get_live_data_v7(exchange):
    # نجلب بيانات أكثر (500 شمعة) لضمان دقة المتوسطات المتحركة
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    bars_4h = exchange.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100)
    df_4h = pd.DataFrame(bars_4h, columns=['ts', 'o', 'h', 'l', 'close_4h', 'v'])
    
    try:
        fund = float(exchange.fetch_funding_rate('BTC/USDT')['fundingRate'])
    except:
        fund = 0.0001
        
    return df, df_4h, fund

def calculate_live_features(df, df_4h, funding_rate):
    """ يجب أن تطابق هذه الدالة دالة التدريب 100% """
    data = df.copy()
    last_4h_close = df_4h.iloc[-1]['close_4h']
    
    data['Returns'] = np.log(data['close'] / data['close'].shift(1))
    data['Range'] = (data['high'] - data['low']) / data['open']
    
    data['Vol_1H'] = data['Returns'].rolling(24).std()
    data['Vol_4H_Proxy'] = data['Returns'].rolling(24).std() # تم التوحيد مع التدريب
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
    
    data['Funding_x_Vol'] = funding_rate * data['Vol_1H']
    
    # Trend 4H: مقارنة الإغلاق الحالي مع آخر إغلاق 4 ساعات
    data['Trend_4H'] = 1 if data['close'].iloc[-1] > last_4h_close else 0
    data['fundingRate'] = funding_rate
    
    return data.iloc[-1]

def run_sniper_v7():
    print(f"\n🔭 (V7 Sniper) Scanning Market... [Threshold: {CONFIDENCE_THRESHOLD}]")
    
    model_path = 'models/btc_v7_ensemble.pkl'
    if not os.path.exists(model_path):
        print("❌ Model not found! Train it first.")
        return

    model = joblib.load(model_path)
    exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        df, df_4h, fund = get_live_data_v7(exchange)
        row = calculate_live_features(df, df_4h, fund)
        
        # ترتيب الميزات ضروري جداً بنفس ترتيب التدريب
        features = [
            'RSI', 'MFI', 'ADX', 'Efficiency_Ratio', 
            'Vol_Ratio', 'CVD_Proxy', 'fundingRate', 
            'Funding_x_Vol', 'Trend_4H', 'Range'
        ]
        
        if row[features].isnull().any():
            print("⚠️ Not enough data for indicators.")
            return

        X_live = pd.DataFrame([row[features]])
        
        # التنبؤ بالاحتمالية
        prob = model.predict_proba(X_live)[0][1]
        
        price = row['close']
        atr = row['ATRr_14'] if 'ATRr_14' in row else price * 0.015 # تقريبي في حال عدم حسابه
        
        print(f"📊 BTC Price: ${price:,.2f} | 🤖 AI Confidence: {prob*100:.2f}%")
        
        # --- منطق اتخاذ القرار ---
        if prob >= CONFIDENCE_THRESHOLD:
            # 1. حساب الأهداف
            sl_dist = atr * 1.5   # توسيع الوقف قليلاً للتقلبات
            tp_dist = atr * 3.0   # الهدف ضعف الوقف (Risk:Reward 1:2)
            
            sl = price - sl_dist
            tp = price + tp_dist
            
            # 2. حساب حجم الصفقة (Risk Management)
            risk_amt = CAPITAL * RISK_PER_TRADE
            # الكمية = المبلغ المخاطر / المسافة للسعر
            position_size_btc = risk_amt / sl_dist
            position_size_usd = position_size_btc * price
            
            msg = (
                f"🔥 **SNIPER SIGNAL DETECTED** 🔥\n"
                f"--------------------------------\n"
                f"🟢 **BUY BTC/USDT**\n"
                f"💵 Price: ${price:,.2f}\n"
                f"🤖 Score: {prob*100:.1f}% (Thresh: {CONFIDENCE_THRESHOLD})\n"
                f"--------------------------------\n"
                f"🛡️ Stop Loss: ${sl:,.2f}\n"
                f"🎯 Take Profit: ${tp:,.2f}\n"
                f"⚖️ Risk/Reward: 1:2.0\n"
                f"💰 Position Size: ${position_size_usd:.2f}\n"
                f"--------------------------------\n"
                f"⚠️ *Enter manually now!*"
            )
            
            send_msg(msg)
            print("✅✅ Signal Sent to Telegram!")
            
        else:
            print("💤 No Trade. Waiting for setup...")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_sniper_v7()