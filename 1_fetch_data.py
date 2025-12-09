import ccxt
import pandas as pd
import os

# إعدادات
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'  # سنعمل على فريم الساعة (ممتاز للسبوت)
LIMIT = 10000     # سنحاول جلب أكبر قدر ممكن من الشموع (تقريباً سنة وشهرين)

def fetch_and_save_data():
    print(f"🔄 جاري الاتصال بـ Binance لجلب بيانات {SYMBOL}...")
    
    exchange = ccxt.binance()
    
    # خدعة لجلب بيانات أكثر من المسموح به في طلب واحد (Pagination)
    # سنكتفي هنا بطلب بسيط لـ 1000 شمعة للتجربة الأولية السريعة
    # لاحقاً سأعلمك كيف تجلب بيانات 5 سنوات
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1000)
    
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # حفظ الملف
    if not os.path.exists('data'):
        os.makedirs('data')
        
    file_path = 'data/btc_data.csv'
    df.to_csv(file_path, index=False)
    
    print(f"✅ تم حفظ {len(df)} شمعة بنجاح في ملف: {file_path}")
    print("👀 نظرة سريعة على البيانات:")
    print(df.tail(3))

if __name__ == "__main__":
    fetch_and_save_data()