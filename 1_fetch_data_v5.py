import ccxt
import pandas as pd
import os
import requests  # سنستخدم هذا لجلب التمويل يدوياً
import time

# إعدادات
SYMBOL_CCXT = 'BTC/USDT'   # الصيغة للمكتبة
SYMBOL_API = 'BTCUSDT'     # الصيغة للرابط المباشر
LIMIT = 1500

def fetch_and_save_data():
    print(f"🚀 (V5) الاتصال بـ Binance (Futures/Swap)...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    try:
        # 1. جلب بيانات 1H (السعر)
        print(f"⏳ جلب بيانات 1H...")
        bars_1h = exchange.fetch_ohlcv(SYMBOL_CCXT, timeframe='1h', limit=LIMIT)
        df_1h = pd.DataFrame(bars_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # تحديد وقت البداية (ملي ثانية)
        start_timestamp = int(df_1h.iloc[0]['timestamp'])
        
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
        
        # 2. جلب بيانات 4H (السياق)
        print(f"⏳ جلب بيانات 4H...")
        bars_4h = exchange.fetch_ohlcv(SYMBOL_CCXT, timeframe='4h', limit=LIMIT // 4)
        df_4h = pd.DataFrame(bars_4h, columns=['timestamp', 'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h'])
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
        
        # 3. جلب Funding Rate (يدوياً عبر API المباشر)
        print("⏳ جلب تاريخ Funding Rate (Direct API)...")
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {
                'symbol': SYMBOL_API,
                'startTime': start_timestamp,
                'limit': 1000  # الحد الأقصى لباينانس
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                df_fund = pd.DataFrame(data)
                # باينانس تعيد fundingTime و fundingRate
                df_fund['timestamp'] = pd.to_datetime(df_fund['fundingTime'], unit='ms')
                df_fund['fundingRate'] = df_fund['fundingRate'].astype(float)
                df_fund = df_fund[['timestamp', 'fundingRate']]
                print(f"   ✅ تم جلب {len(df_fund)} سجل تمويل بنجاح.")
            else:
                raise ValueError("البيانات العائدة فارغة")

        except Exception as e:
            print(f"⚠️ فشل جلب التمويل ({e}) - استخدام قيم افتراضية.")
            df_fund = df_1h[['timestamp']].copy()
            df_fund['fundingRate'] = 0.0001 

        # --- الدمج ---
        print("⚗️ دمج البيانات...")
        
        # دمج 1H مع 4H
        df_merged = pd.merge_asof(df_1h.sort_values('timestamp'), 
                                  df_4h.sort_values('timestamp'), 
                                  on='timestamp', 
                                  direction='backward')
        
        # دمج Funding
        df_final = pd.merge_asof(df_merged, 
                                 df_fund.sort_values('timestamp'), 
                                 on='timestamp', 
                                 direction='backward')
        
        # تعبئة البيانات المفقودة
        df_final.ffill(inplace=True) 
        df_final.dropna(inplace=True)
        
        if not os.path.exists('data'):
            os.makedirs('data')
        
        file_path = 'data/btc_data_v5.csv'
        df_final.to_csv(file_path, index=False)
        
        print(f"✅ تم الحفظ بنجاح! ({len(df_final)} شمعة)")
        print(f"   عينة التمويل: {df_final['fundingRate'].iloc[-1]}")
        print("➡️ الآن البيانات جاهزة تماماً. شغل الملف 2.")

    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    fetch_and_save_data()