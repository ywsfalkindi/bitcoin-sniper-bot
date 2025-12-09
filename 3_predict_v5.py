import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_msg(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params)
    except Exception as e:
        print(f"⚠️ Telegram Error: {e}")

def get_market_sentiment_v5():
    print("🛰️ (V5) القناص يعمل: تحليل السوق المباشر...")
    
    model_path = 'models/btc_v5_worldclass.pkl'
    if not os.path.exists(model_path):
        print("❌ ملف النموذج غير موجود! درب النموذج أولاً (الملف 2).")
        return

    # --- تصحيح الاتصال هنا ---
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    try:
        exchange.load_markets()
        # تحديد الرمز الصحيح تلقائياً
        symbol = 'BTC/USDT'
        if symbol not in exchange.markets:
            for m in exchange.markets:
                if m.startswith('BTC/USDT'):
                    symbol = m
                    break
        
        # 1. جلب البيانات الحية
        bars_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        bars_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=50)
        
        # جلب Funding Rate الحالي
        try:
            funding_info = exchange.fetch_funding_rate(symbol)
            current_funding = funding_info['fundingRate']
        except:
            current_funding = 0.0001 # قيمة افتراضية
        
        # تحويل لـ DataFrame
        df = pd.DataFrame(bars_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h = pd.DataFrame(bars_4h, columns=['timestamp', 'o', 'h', 'l', 'close_4h', 'v'])
        
        # تجهيز مؤشرات 4H (السياق)
        last_close_4h = df_4h['close_4h'].iloc[-1]
        mean_4h_50 = df_4h['close_4h'].rolling(50).mean().iloc[-1]
        if pd.isna(mean_4h_50): mean_4h_50 = last_close_4h # حماية من القيم الفارغة في البداية
        
        # --- هندسة الميزات (نفس التدريب) ---
        df['RSI'] = df.ta.rsi(length=14)
        df['EMA_50'] = df.ta.ema(length=50)
        df['Trend_1H'] = (df['close'] > df['EMA_50']).astype(int)
        
        # دمج ميزات 4H والتمويل
        df['Trend_4H'] = 1 if last_close_4h > mean_4h_50 else 0
        df['RSI_4H_Divergence'] = df['close'] / last_close_4h
        
        df['ATR'] = df.ta.atr(length=14)
        df['ATR_Pct'] = df['ATR'] / df['close']
        df['Force_Index'] = df['close'].diff(1) * df['volume']
        
        df['fundingRate'] = current_funding
        df['Funding_Risk'] = 1 if current_funding > 0.01 else 0
        
        # أخذ آخر صف مكتمل
        current = df.iloc[-1]
        
        # --- التنبؤ ---
        model = joblib.load(model_path)
        features = [
            'RSI', 'Trend_1H', 'Trend_4H', 'ATR_Pct', 
            'fundingRate', 'Funding_Risk', 'Force_Index', 'RSI_4H_Divergence'
        ]
        
        # التأكد من عدم وجود قيم NaN قبل التنبؤ
        if current[features].isnull().any():
            print("⚠️ البيانات غير كافية لحساب المؤشرات حالياً.")
            return

        input_data = pd.DataFrame([current[features]])
        
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        
        price = current['close']
        atr = current['ATR']
        
        print(f"📊 السعر: ${price:,.2f} | الثقة: {prob:.2f}% | التمويل: {current_funding*100:.4f}%")
        
        # --- نظام الفيتو ---
        veto = False
        if current['RSI'] > 75: veto = True
        if current_funding > 0.02: veto = True
            
        # إرسال التنبيه
        if pred == 1 and prob > 65 and not veto:
            sl = price - (atr * 1.5)
            tp = price + (atr * 3.0)
            msg = (
                f"🔥 **إشارة قناص V5** 🔥\n"
                f"💎 **BTC/USDT**\n"
                f"السعر: ${price:,.2f}\n"
                f"الهدف: ${tp:,.2f}\n"
                f"الوقف: ${sl:,.2f}\n"
                f"الثقة: {prob:.1f}%\n"
                f"التمويل: {current_funding*100:.4f}%"
            )
            send_msg(msg)
            print("🚀 تم إرسال التوصية!")
        else:
            print("😴 لا توجد فرصة قوية الآن.")

    except Exception as e:
        print(f"❌ خطأ أثناء التنبؤ: {e}")

if __name__ == "__main__":
    get_market_sentiment_v5()