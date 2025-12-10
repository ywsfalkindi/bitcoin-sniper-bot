import time
import os
import sys
import pandas as pd
import joblib
import ccxt
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. هندسة الميزات (نفس نسخة التدريب بالضبط)
# ==========================================
def feature_engineering_v9(df):
    data = df.copy()
    
    # 1. ضبط الفهرس
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data.sort_index(inplace=True)
            
    # 2. الحسابات الرياضية
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    
    data['GK_Vol'] = ((np.log(data['high'] / data['low'])**2) / 2) - \
                     (2 * np.log(2) - 1) * ((np.log(data['close'] / data['open'])**2))
    
    # 3. المؤشرات
    data['RSI'] = data.ta.rsi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14']
    data['ATR'] = data.ta.atr(length=14)
    
    # VWAP مع حماية
    data['vwap'] = data.ta.vwap()
    if data['vwap'].isnull().all():
         data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['dist_vwap'] = (data['close'] - data['vwap']) / (data['vwap'] + 1e-9)
    
    # تعبئة القيم الفارغة
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    
    return data

# ==========================================
# 2. وظيفة التوقع
# ==========================================
def get_latest_data_and_predict():
    try:
        print("\n🔄 جاري فحص السوق...")
        
        # تحميل ملف النموذج
        if not os.path.exists('models/btc_v9_worldclass.pkl'):
            print("❌ ملف النموذج غير موجود!")
            return

        packet = joblib.load('models/btc_v9_worldclass.pkl')
        
        # فك الصندوق (استخراج النموذج والعتبة)
        if isinstance(packet, dict):
            model = packet['model']
            THRESHOLD = packet['threshold']
            print(f"⚙️ النموذج نشط | العتبة الذكية: {THRESHOLD:.2f}")
        else:
            model = packet
            THRESHOLD = 0.62 # قيمة احتياطية
            print(f"⚙️ نموذج قديم | العتبة الافتراضية: {THRESHOLD}")

        # جلب البيانات الحية
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # معالجة البيانات
        df = feature_engineering_v9(df)
        
        # تجهيز الصف الأخير
        last_row = df.iloc[-1:]
        
        # التأكد من وجود الميزات المطلوبة
        features = ['log_ret', 'GK_Vol', 'dist_vwap', 'RSI', 'ADX', 'ATR']
        
        # إضافة fng_value إذا كان النموذج يحتاجها (قيمة افتراضية)
        if hasattr(model, 'feature_names_in_') and 'fng_value' in model.feature_names_in_:
            last_row['fng_value'] = 50 
            
        X_live = last_row[features]
        
        # التوقع
        prob = model.predict_proba(X_live)[0][1]
        price = last_row['close'].values[0]
        timestamp = last_row.index[0]
        
        print(f"⏱️ {timestamp} | 💰 Price: {price}")
        print(f"🔮 احتمالية الصعود: {prob:.2%} (المطلوب: {THRESHOLD:.2%})")
        
        if prob >= THRESHOLD:
            print("🚀 ✅ إشارة شراء قوية (SNIPER ENTRY)!")
            # exchange.create_market_buy_order('BTC/USDT', amount)
        else:
            print("💤 السوق غير مناسب للدخول.")
            
    except Exception as e:
        print(f"⚠️ خطأ: {e}")

# ==========================================
# 3. حلقة التشغيل
# ==========================================
def wait_for_next_candle():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds()
    print(f"⏳ الانتظار {wait_seconds/60:.1f} دقيقة حتى إغلاق الشمعة...")
    time.sleep(wait_seconds + 5) 

def main():
    print("💎 BTC V9.1 SNIPER - LIVE TRADING")
    # فحص أولي فوري
    get_latest_data_and_predict()
    
    while True:
        wait_for_next_candle()
        get_latest_data_and_predict()

if __name__ == "__main__":
    main()