import ccxt
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import os
import requests  # المكتبة الجديدة للإرسال
from dotenv import load_dotenv  # استدعاء المكتبة

load_dotenv()

# ==========================================
# ⚙️ إعدادات تيليجرام (ضع بياناتك هنا)
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_msg(text):
    """دالة لإرسال الرسالة إلى تيليجرام"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params)
        print("📨 تم إرسال التنبيه إلى تيليجرام.")
    except Exception as e:
        print(f"⚠️ فشل إرسال الرسالة: {e}")

def get_advice_v3():
    model_path = 'models/btc_v3_smart.json'
    if not os.path.exists(model_path):
        print("❌ لم يتم العثور على النموذج V3!")
        return

    print("⏳ جاري تحليل السوق وإعداد التقرير...")
    
    # 1. جلب البيانات
    exchange = ccxt.binance()
    try:
        bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
    except:
        print("⚠️ مشكلة في الاتصال بالإنترنت")
        return

    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 2. الحسابات (نفس التدريب V3)
    df['returns'] = df['close'].pct_change()
    df['EMA_50'] = df.ta.ema(length=50)
    df['dist_EMA50'] = (df['close'] / df['EMA_50']) - 1
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR_Pct'] = df.ta.atr(length=14) / df['close']
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Whale_Activity'] = df['volume'] / df['Vol_MA20']
    
    bb = df.ta.bbands(length=20, std=2)
    upper = bb.iloc[:, 2]
    lower = bb.iloc[:, 0]
    df['BB_Width'] = (upper - lower) / df['close']
    
    current = df.iloc[[-1]]
    
    # 3. التنبؤ
    model = XGBClassifier()
    model.load_model(model_path)
    
    features = ['returns', 'dist_EMA50', 'RSI', 'ATR_Pct', 'Whale_Activity', 'BB_Width']
    pred = model.predict(current[features])[0]
    prob = model.predict_proba(current[features])[0][1]
    
    # 4. تجهيز الرسالة
    price = current['close'].values[0]
    whale = current['Whale_Activity'].values[0]
    prob_perc = prob * 100
    
    print(f"💰 السعر: ${price:,.2f} | الثقة: {prob_perc:.2f}%")

    # --- منطق الإرسال الذكي ---
    # نرسل رسالة في حالتين فقط:
    # 1. إذا كان القرار شراء (Buy)
    # 2. أو إذا كانت الثقة مرتفعة نسبياً (فوق 30%) حتى لو لم تكن شراء، للتنبيه
    
    if pred == 1:
        msg = (
            f"🚀 **إشارة شراء قوية (STRONG BUY)**\n"
            f"--------------------------------\n"
            f"💰 السعر: ${price:,.2f}\n"
            f"📊 الثقة: {prob_perc:.2f}%\n"
            f"🐋 نشاط الحيتان: {whale:.2f}x\n"
            f"⏰ الوقت: {pd.Timestamp.now().strftime('%H:%M')}\n"
            f"--------------------------------\n"
            f"💡 *النصيحة:* فرصة دخول ممتازة بناءً على تحركات الحيتان."
        )
        send_msg(msg) # أرسل فوراً
        print("🚀 القرار: شراء (تم الإرسال)")
        
    elif prob_perc > 30: # (اختياري) تنبيه عند بدء تحسن السوق
        msg = (
            f"👀 **تنبيه: السوق بدأ يتحرك**\n"
            f"السعر: ${price:,.2f}\n"
            f"احتمالية الصعود ارتفعت إلى: {prob_perc:.2f}%\n"
            f"لا يوجد قرار شراء مؤكد بعد، لكن كن مستعداً."
        )
        send_msg(msg)
        print("👀 تنبيه مبدئي (تم الإرسال)")
        
    else:
        print("✋ القرار: انتظر (لن يتم إرسال رسالة لتجنب الإزعاج)")

if __name__ == "__main__":
    # تجربة إرسال رسالة ترحيبية عند التشغيل للتأكد
    send_msg("🤖 تم تشغيل بوت القناص بنجاح!")
    get_advice_v3()