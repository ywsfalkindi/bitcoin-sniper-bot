import ccxt
import pandas as pd
import os

# إعدادات
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 2000  # زدنا العدد لتعلم أنماط أكثر

def fetch_and_save_data():
    print(f"🔄 (V4) جاري الاتصال لجلب بيانات {SYMBOL} الاحترافية...")
    
    exchange = ccxt.binance()
    
    # جلب البيانات
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    except Exception as e:
        print(f"❌ خطأ في الاتصال: {e}")
        return
    
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # حفظ الملف
    if not os.path.exists('data'):
        os.makedirs('data')
        
    file_path = 'data/btc_data.csv'
    df.to_csv(file_path, index=False)
    
    print(f"✅ تم حفظ {len(df)} شمعة. البيانات جاهزة للتدريب الذكي.")

if __name__ == "__main__":
    fetch_and_save_data()