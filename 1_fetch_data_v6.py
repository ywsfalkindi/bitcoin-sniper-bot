import ccxt
import pandas as pd
import os
import requests
import time
import numpy as np

# إعدادات
SYMBOL_CCXT = 'BTC/USDT'
SYMBOL_API = 'BTCUSDT'
LIMIT = 2000 # زيادة البيانات لتدريب أفضل

def fetch_data_v6():
    print(f"🚀 (V6 Architect) بدء سحب البيانات العميقة لـ {SYMBOL_CCXT}...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    try:
        # 1. جلب بيانات 1H (الأساسي)
        print(f"⏳ جلب الشموع الساعية (1H)...")
        bars_1h = exchange.fetch_ohlcv(SYMBOL_CCXT, timeframe='1h', limit=LIMIT)
        # ملاحظة: Binance تعيد Taker buy base asset volume في العمود السادس غالباً، لكن CCXT يوحدها
        # سنعتمد على Volume وسنحسب الـ Buying Pressure برمجياً
        df_1h = pd.DataFrame(bars_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        start_time = int(df_1h.iloc[0]['timestamp'])
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')

        # 2. جلب بيانات 4H (السياق والاتجاه العام)
        print(f"⏳ جلب شموع السياق (4H)...")
        bars_4h = exchange.fetch_ohlcv(SYMBOL_CCXT, timeframe='4h', limit=LIMIT // 2)
        df_4h = pd.DataFrame(bars_4h, columns=['timestamp', 'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h'])
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')

        # 3. جلب Funding Rate (مؤشر الخوف والطمع المؤسسي)
        print("⏳ جلب بيانات Funding Rate التاريخية...")
        funding_data = []
        # سنحاول جلب أكبر قدر ممكن (Binance API limits apply)
        end_time = int(time.time() * 1000)
        # جلب آخر 1000 نقطة فقط للسرعة، في الإنتاج الفعلي تحتاج لتجميع مستمر
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {'symbol': SYMBOL_API, 'limit': 1000}
        
        try:
            resp = requests.get(url, params=params).json()
            if isinstance(resp, list):
                for x in resp:
                    funding_data.append({
                        'timestamp': pd.to_datetime(x['fundingTime'], unit='ms'),
                        'fundingRate': float(x['fundingRate'])
                    })
                df_fund = pd.DataFrame(funding_data)
            else:
                raise ValueError("Format Error")
        except:
            print("⚠️ تعذر جلب التمويل، استخدام القيم الصفرية.")
            df_fund = pd.DataFrame({'timestamp': df_1h['timestamp'], 'fundingRate': 0.0001})

        # --- الدمج الذكي (Merge Asof) ---
        print("⚗️ معالجة ودمج الجداول الزمنية...")
        df_merged = pd.merge_asof(df_1h.sort_values('timestamp'), 
                                  df_4h.sort_values('timestamp'), 
                                  on='timestamp', direction='backward')
        
        df_final = pd.merge_asof(df_merged, 
                                 df_fund.sort_values('timestamp'), 
                                 on='timestamp', direction='backward')
        
        # تنظيف
        df_final.ffill(inplace=True)
        df_final.dropna(inplace=True)

        if not os.path.exists('data'): os.makedirs('data')
        df_final.to_csv('data/btc_data_v6.csv', index=False)
        print(f"✅ تم الحفظ: data/btc_data_v6.csv ({len(df_final)} صف)")

    except Exception as e:
        print(f"❌ خطأ قاتل: {e}")

if __name__ == "__main__":
    fetch_data_v6()