import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_brain_v3():
    print("🧠 (V3) تدريب النموذج 'المعمم' (Genaralized Model)...")
    print("هدفي الآن: تعلم الأنماط وليس حفظ الأسعار.")
    
    if not os.path.exists('data/btc_data.csv'):
        print("❌ شغل 1_fetch_data.py أولاً")
        return
    df = pd.read_csv('data/btc_data.csv')
    
    # --- هندسة الميزات الذكية (بدون أرقام خام) ---
    
    # 1. التغير السعري (بدل السعر نفسه)
    # هل السعر صاعد أم هابط مقارنة بالساعة الماضية؟
    df['returns'] = df['close'].pct_change()
    
    # 2. المسافة عن المتوسطات (Distance to EMAs)
    # بدل أن نقول السعر 90 ألف، نقول: السعر أعلى من المتوسط بـ 2%
    df['EMA_50'] = df.ta.ema(length=50)
    df['dist_EMA50'] = (df['close'] / df['EMA_50']) - 1
    
    # 3. المؤشرات النسبية (هي أصلاً نسب مئوية فلا خوف منها)
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR_Pct'] = df.ta.atr(length=14) / df['close'] # الـ ATR كنسبة من السعر
    
    # 4. كاشف الحيتان (نسبي أيضاً)
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Whale_Activity'] = df['volume'] / df['Vol_MA20']
    
    # 5. عرض البولنجر (نسبي)
    bb = df.ta.bbands(length=20, std=2)
    upper = bb.iloc[:, 2]
    lower = bb.iloc[:, 0]
    df['BB_Width'] = (upper - lower) / df['close']

    # تنظيف
    df.dropna(inplace=True)
    
    # --- تحديد الهدف ---
    # نشتري إذا ارتفع السعر 1.5% خلال 12 ساعة
    FUTURE_PERIOD = 12
    PROFIT_TARGET = 0.015
    df['future_close'] = df['close'].shift(-FUTURE_PERIOD)
    df['Target'] = (df['future_close'] > df['close'] * (1 + PROFIT_TARGET)).astype(int)
    df.dropna(inplace=True)
    
    # --- اختيار الميزات (لاحظ: حذفنا open, high, low, close, volume) ---
    features = ['returns', 'dist_EMA50', 'RSI', 'ATR_Pct', 'Whale_Activity', 'BB_Width']
    X = df[features]
    y = df['Target']
    
    # تقسيم زمني (غير مخلوط)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # التدريب
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        scale_pos_weight=3 # التركيز على اقتناص الفرص النادرة
    )
    model.fit(X_train, y_train)
    
    # التقييم
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n📊 دقة النموذج المعمم: {acc*100:.2f}%")
    
    # ما هو أهم عامل الآن؟ (المفاجأة)
    scores = model.get_booster().get_score(importance_type='weight')
    print(f"💡 العامل الأهم في القرار: {max(scores, key=scores.get)}")
    
    model.save_model('models/btc_v3_smart.json')
    print("✅ تم حفظ النموذج الذكي.")

if __name__ == "__main__":
    train_brain_v3()