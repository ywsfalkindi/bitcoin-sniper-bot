import ccxt
import pandas as pd
import os
import time
from datetime import datetime
from tqdm import tqdm  # شريط تقدم
import numpy as np

# إعدادات
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
START_DATE = "2020-01-01 00:00:00"  # سنبدأ من 2020 لتدريب النموذج على كل ظروف السوق

def fetch_ohlcv_history(exchange, symbol, timeframe, start_str):
    """ دالة لجلب التاريخ كاملاً عبر التجزئة (Pagination) """
    
    # تحويل التاريخ إلى Timestamp
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    print(f"⏳ جلب بيانات {symbol} من {start_str} حتى الآن...")
    
    # تقدير عدد الشموع المتوقعة لشريط التقدم
    now = exchange.milliseconds()
    total_time = now - since
    # 1h = 3600000 ms
    estimated_candles = total_time / 3600000
    
    pbar = tqdm(total=int(estimated_candles), unit=" candle")
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            if len(ohlcv) == 0:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # تحديث الوقت لآخر شمعة + 1
            last_time = ohlcv[-1][0]
            since = last_time + 1
            
            pbar.update(len(ohlcv))
            
            # استراحة صغيرة لتجنب الحظر
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️ خطأ في الاتصال: {e}")
            time.sleep(2)
            continue
            
    pbar.close()
    return all_ohlcv

def fetch_funding_history_v8(symbol_api, start_ms):
    """ جلب التمويل المتوافق مع الفترة الزمنية """
    import requests
    funding_data = []
    current_time = start_ms
    end_time = int(time.time() * 1000)
    
    print("\n⏳ جلب تاريخ Funding Rate (مؤشر المؤسسات)...")
    pbar = tqdm(total=(end_time - current_time), unit='ms')
    
    while current_time < end_time:
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {'symbol': symbol_api, 'startTime': current_time, 'limit': 1000}
            resp = requests.get(url, params=params, timeout=10).json()
            
            if not resp: break
            
            for x in resp:
                funding_data.append({
                    'timestamp': pd.to_datetime(x['fundingTime'], unit='ms'),
                    'fundingRate': float(x['fundingRate'])
                })
            
            last_ts = resp[-1]['fundingTime']
            if last_ts == current_time: current_time += 3600000 * 8 # تجاوز
            else: current_time = last_ts + 1
            
            pbar.update(last_ts - current_time)
            time.sleep(0.1)
            
        except:
            time.sleep(1)
            
    pbar.close()
    return pd.DataFrame(funding_data)

def main_v8():
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # 1. جلب الشموع التاريخية (1H)
    data = fetch_ohlcv_history(exchange, SYMBOL, TIMEFRAME, START_DATE)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"✅ تم تحميل {len(df)} شمعة (ساعة).")
    
    # 2. جلب Funding Rate لنفس الفترة
    start_ms = int(df.iloc[0]['timestamp'].timestamp() * 1000)
    df_fund = fetch_funding_history_v8('BTCUSDT', start_ms)
    
    # 3. جلب السياق (4H) - سنقوم ببنائه برمجياً من 1H لضمان التوافق التام
    # (Resampling is better for historical data consistency)
    print("🔄 إعادة تشكيل بيانات 4H من البيانات الأساسية...")
    df.set_index('timestamp', inplace=True)
    
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h.columns = [f"{c}_4h" for c in df_4h.columns]
    
    df.reset_index(inplace=True)
    df_4h.reset_index(inplace=True)
    
    # 4. الدمج النهائي
    print("⚗️ دمج كل البيانات...")
    df_merged = pd.merge_asof(df, df_4h, on='timestamp', direction='backward')
    
    if not df_fund.empty:
        df_final = pd.merge_asof(df_merged, df_fund.sort_values('timestamp'), on='timestamp', direction='backward')
    else:
        df_final = df_merged
        df_final['fundingRate'] = 0.0001
        
    df_final.ffill(inplace=True)
    df_final.dropna(inplace=True)
    
    if not os.path.exists('data'): os.makedirs('data')
    df_final.to_csv('data/btc_data_v7.csv', index=False) # نحتفظ بنفس الاسم ليعمل ملف التدريب
    print(f"\n🎉 تم بناء قاعدة بيانات عملاقة: {len(df_final)} سجل.")
    print("الآن نموذجك سيرى كل شيء: كورونا، قمة 69k، قاع 15k، والوضع الحالي!")

if __name__ == "__main__":
    main_v8()