import ccxt
import pandas as pd
import numpy as np
import requests
import time
import os
from tqdm import tqdm
from datetime import datetime

# إعدادات
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
START_DATE = "2020-01-01 00:00:00"

def fetch_fear_and_greed(start_date):
    """ جلب بيانات الخوف والطمع كمؤشر للمشاعر (Sentiment) """
    print("🧠 جلب بيانات Fear & Greed Index...")
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        response = requests.get(url).json()
        data = response['data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['fng_value'] = df['value'].astype(int)
        df = df[['timestamp', 'fng_value']]
        # مواءمة التوقيت ليكون بالساعة (لأن المؤشر يومي)
        df.set_index('timestamp', inplace=True)
        df = df.resample('1h').ffill()
        return df
    except Exception as e:
        print(f"⚠️ تعذر جلب بيانات المشاعر: {e}")
        return pd.DataFrame()

def fetch_ohlcv_advanced(exchange, symbol, timeframe, start_str):
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    print(f"⏳ جلب بيانات السوق والسيولة لـ {symbol}...")
    
    now = exchange.milliseconds()
    pbar = tqdm(total=int((now - since) / 3600000))
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            pbar.update(len(ohlcv))
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            
    pbar.close()
    return all_ohlcv

def main_v9():
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # 1. البيانات الأساسية
    data = fetch_ohlcv_advanced(exchange, SYMBOL, TIMEFRAME, START_DATE)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 2. جلب Funding Rate (مهم جداً للمؤسسات)
    # سنستخدم تقريب هنا لتسريع الكود، في الإنتاج استخدم الدالة السابقة
    print("💸 جلب معدلات التمويل (Funding Rates)...")
    # (تم تبسيط هذا الجزء ليعمل مباشرة، يفضل استخدام دالة v8 السابقة للتمويل الدقيق)
    # هنا سنفترض قيمة افتراضية للسرعة، استبدلها ببيانات حقيقية إذا توفرت
    df['fundingRate'] = 0.0001 
    
    # 3. جلب بيانات المشاعر
    df_fng = fetch_fear_and_greed(START_DATE)
    
    # 4. دمج البيانات
    df.set_index('timestamp', inplace=True)
    
    # دمج المشاعر
    if not df_fng.empty:
        df = df.join(df_fng, how='left')
        df['fng_value'] = df['fng_value'].fillna(method='ffill')
    else:
        df['fng_value'] = 50 # محايد
        
    # بناء بيانات 4 ساعات للسياق
    df_4h = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    df_4h.columns = [f"{c}_4h" for c in df_4h.columns]
    
    df = pd.merge_asof(df.sort_index(), df_4h.sort_index(), left_index=True, right_index=True, direction='backward')
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    
    if not os.path.exists('data'): os.makedirs('data')
    df.to_csv('data/btc_data_v9.csv', index=False)
    print(f"✅ تم إنشاء قاعدة بيانات V9 المتطورة: {len(df)} سجل.")

if __name__ == "__main__":
    main_v9()