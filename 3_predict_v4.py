import ccxt
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import os
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_msg(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
        requests.get(url, params=params)
        print("📨 تم إرسال التنبيه.")
    except Exception as e:
        print(f"⚠️ فشل الإرسال: {e}")

def get_advice_v4():
    model_path = 'models/btc_v4_sniper.json'
    if not os.path.exists(model_path):
        print("❌ لم يتم العثور على النموذج V4!")
        return

    print("⏳ (V4) تحليل السوق وحساب المخاطر...")
    
    # 1. جلب بيانات حية
    exchange = ccxt.binance()
    try:
        bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
    except:
        print("⚠️ مشكلة انترنت")
        return

    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 2. تجهيز المؤشرات (نفس التدريب تماماً)
    df['returns'] = df['close'].pct_change()
    df['EMA_50'] = df.ta.ema(length=50)
    df['dist_EMA50'] = (df['close'] / df['EMA_50']) - 1
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14) # نحتاج القيمة الخام لحساب الأهداف
    df['ATR_Pct'] = df['ATR'] / df['close']
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Whale_Activity'] = df['volume'] / df['Vol_MA20']
    
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close']
    
    current = df.iloc[[-1]] # آخر شمعة
    
    # 3. التنبؤ
    model = XGBClassifier()
    model.load_model(model_path)
    
    features = ['returns', 'dist_EMA50', 'RSI', 'ATR_Pct', 'Whale_Activity', 'BB_Width']
    pred = model.predict(current[features])[0]
    prob = model.predict_proba(current[features])[0][1] * 100
    
    # 4. حساب أرقام الصفقة (إدارة المخاطر)
    price = current['close'].values[0]
    atr_val = current['ATR'].values[0]
    
    # استراتيجية: وقف الخسارة أسفل السعر بـ 1.5 ATR، والهدف 2 ATR
    stop_loss = price - (atr_val * 1.5)
    take_profit = price + (atr_val * 2.5)
    risk_reward = (take_profit - price) / (price - stop_loss)
    
    whale = current['Whale_Activity'].values[0]
    
    print(f"💰 السعر: ${price:,.2f} | الثقة: {prob:.2f}%")

    # 5. منطق الإرسال الذكي
    if pred == 1 and prob > 55: # شرط قوي
        msg = (
            f"🔥 **إشارة قناص مؤكدة (V4)** 🔥\n"
            f"--------------------------------\n"
            f"💎 **العملة:** #BTC/USDT\n"
            f"💵 **الدخول:** ${price:,.2f}\n"
            f"--------------------------------\n"
            f"🛑 **Stop Loss:** ${stop_loss:,.2f}\n"
            f"🎯 **Target:** ${take_profit:,.2f}\n"
            f"⚖️ **R/R Ratio:** {risk_reward:.2f}\n"
            f"--------------------------------\n"
            f"📊 الثقة: {prob:.2f}%\n"
            f"🐋 نشاط الحيتان: {whale:.2f}x\n"
        )
        send_msg(msg)
        print("🚀 تم إرسال صفقة شراء كاملة!")
        
    elif prob > 35:
        msg = (
            f"👀 **تنبيه مراقبة**\n"
            f"السعر: ${price:,.2f}\n"
            f"الثقة: {prob:.2f}%\n"
            f"السوق يتحسن، انتظر إشارة الدخول."
        )
        send_msg(msg)
    else:
        print("😴 السوق غير مناسب للدخول.")

if __name__ == "__main__":
    get_advice_v4()