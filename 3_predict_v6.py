import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# إعدادات المخاطر
RISK_PER_TRADE = 0.02 # نخاطر بـ 2% من المحفظة
CAPITAL = 1000 # محفظة افتراضية للحساب (أو اجلب الرصيد الحقيقي)

def send_msg(text):
    if not TELEGRAM_TOKEN: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except: pass

def get_live_data(exchange):
    symbol = 'BTC/USDT'
    # جلب 1H
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # جلب 4H
    bars_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=50)
    df_4h = pd.DataFrame(bars_4h, columns=['ts', 'o', 'h', 'l', 'close_4h', 'v'])
    
    # جلب التمويل
    try:
        fund = float(exchange.fetch_funding_rate(symbol)['fundingRate'])
    except:
        fund = 0.0001
        
    return df, df_4h, fund

def calculate_features_live(df, df_4h, funding_rate):
    # نفس دالة التدريب تماماً لضمان الاتساق
    data = df.copy()
    
    # دمج آخر إغلاق 4H
    last_4h = df_4h.iloc[-1]['close_4h']
    
    data['HA_Close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['HA_Open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2 # تقريبي للبيانات الحية
    
    data['RSI'] = data.ta.rsi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14']
    
    data['Log_Ret'] = np.log(data['close'] / data['close'].shift(1))
    data['Volatility'] = data['Log_Ret'].rolling(window=24).std() * np.sqrt(24)
    data['ATR'] = data.ta.atr(length=14)
    
    data['Z_Score'] = (data['close'] - data['close'].rolling(20).mean()) / (data['close'].rolling(20).std() + 1e-9)
    
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    
    data['Buying_Pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-9) * data['volume']
    
    data['Trend_4H'] = 1 if data['close'].iloc[-1] > last_4h else 0
    data['Divergence'] = data['close'] / last_4h
    data['fundingRate'] = funding_rate
    
    return data.iloc[-1]

def kelly_criterion(win_prob, win_loss_ratio=2.5):
    """ حساب حجم الصفقة المثالي """
    # f = (p(b+1) - 1) / b
    # p = win probability, b = win/loss ratio
    kelly = (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio
    return max(0, kelly * 0.5) # نستخدم نصف كيلي للأمان

def run_sniper_v6():
    print("🛰️ (V6 Sniper) تحليل السوق...")
    model_path = 'models/btc_v6_model.pkl'
    if not os.path.exists(model_path): return
    
    model = joblib.load(model_path)
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    
    try:
        df, df_4h, fund = get_live_data(exchange)
        row = calculate_features_live(df, df_4h, fund)
        
        # تجهيز المدخلات
        features = [
            'RSI', 'ADX', 'Z_Score', 'Volatility', 
            'Buying_Pressure', 'fundingRate', 
            'hour_sin', 'hour_cos', 'Trend_4H', 'Divergence'
        ]
        
        if row[features].isnull().any():
            print("⚠️ البيانات غير مكتملة للمؤشرات.")
            return

        X_live = pd.DataFrame([row[features]])
        
        # التنبؤ
        prob = model.predict_proba(X_live)[0][1] # احتمال الصعود
        
        price = row['close']
        atr = row['ATR']
        adx = row['ADX']
        
        print(f"📊 السعر: {price:.1f} | احتمال الصعود: {prob*100:.1f}% | ADX: {adx:.1f}")
        
        # --- الفلاتر العالمية (World Class Filters) ---
        # 1. فلتر الاتجاه: لا تداول إذا السوق ميت (ADX < 20)
        if adx < 20:
            print("😴 السوق عرضي وممل (ADX منخفض). لا صفقات.")
            return
            
        # 2. فلتر الاحتمال العالي
        if prob > 0.70: # نحتاج ثقة عالية جداً
            # حسابات المخاطرة
            sl = price - (atr * 1.2) # وقف خسارة ديناميكي
            tp1 = price + (atr * 2.0)
            tp2 = price + (atr * 4.0)
            
            # حساب حجم المركز (Position Size)
            # المسافة للوقف %
            dist_sl_pct = (price - sl) / price
            # المبلغ المعرض للخطر = CAPITAL * RISK_PER_TRADE
            # حجم الصفقة = Risk Amount / Distance %
            position_size_usd = (CAPITAL * RISK_PER_TRADE) / dist_sl_pct
            
            # كيلي للتحقق (اختياري)
            kelly_factor = kelly_criterion(prob)
            
            msg = (
                f"🚀 **إشارة V6 المؤكدة** 🚀\n"
                f"🟢 **LONG BTC/USDT**\n"
                f"السعر: ${price:,.2f}\n\n"
                f"🛑 الوقف: ${sl:,.2f}\n"
                f"🎯 هدف 1: ${tp1:,.2f}\n"
                f"🎯 هدف 2: ${tp2:,.2f}\n\n"
                f"🧠 الثقة: {prob*100:.1f}%\n"
                f"💪 قوة الاتجاه (ADX): {adx:.1f}\n"
                f"💰 حجم الصفقة المقترح: ${position_size_usd:.0f} (Leverage x5)"
            )
            send_msg(msg)
            print("🔥 تم إرسال الإشارة!")
        else:
            print("👀 نراقب بصمت... الفرصة لم تكتمل.")
            
    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    run_sniper_v6()