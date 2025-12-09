import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def train_brain_v2():
    print("🧠 (V2) بدء تدريب صائد الحيتان...")
    
    # 1. تحميل البيانات
    if not os.path.exists('data/btc_data.csv'):
        print("❌ ملف البيانات غير موجود! شغل 1_fetch_data.py أولاً")
        return
    df = pd.read_csv('data/btc_data.csv')
    
    # 2. هندسة الميزات المتقدمة (Whale Features) 🐋
    
    # أ) المؤشرات التقليدية
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    
    # ب) كاشف الحيتان (Volume Shock)
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Whale_Activity'] = df['volume'] / df['Vol_MA20']
    
    # ج) كاشف التجميع (Consolidation Squeeze) - التعديل هنا ✅
    # نحسب البولنجر باندز
    bb = df.ta.bbands(length=20, std=2)
    # نستخدم iloc لجلب العمود الأول (السفلي) والثالث (العلوي) بغض النظر عن الاسم
    # العمود 0 = Lower, العمود 1 = Mid, العمود 2 = Upper
    upper_band = bb.iloc[:, 2]
    lower_band = bb.iloc[:, 0]
    
    df['BB_Width'] = (upper_band - lower_band) / df['close']
    
    # تنظيف البيانات
    df.dropna(inplace=True)
    
    # 3. تحديد الهدف (Target)
    FUTURE_PERIOD = 24
    PROFIT_TARGET = 0.020 
    
    df['future_close'] = df['close'].shift(-FUTURE_PERIOD)
    df['Target'] = (df['future_close'] > df['close'] * (1 + PROFIT_TARGET)).astype(int)
    df.dropna(inplace=True)
    
    # 4. التجهيز
    features = ['close', 'volume', 'RSI', 'EMA_50', 'EMA_200', 'Whale_Activity', 'BB_Width']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. تدريب النموذج
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        random_state=42,
        scale_pos_weight=3
    )
    
    model.fit(X_train, y_train)
    
    # 6. التقييم
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    feature_important = model.get_booster().get_score(importance_type='weight')
    
    print(f"\n📊 دقة النموذج الجديد: {accuracy * 100:.2f}%")
    if feature_important:
        print(f"💡 أهم عامل يعتمد عليه النموذج حالياً: {max(feature_important, key=feature_important.get)}")
    else:
        print("💡 لم يتم تحديد العامل الأهم بعد.")
    
    # حفظ النموذج
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_model('models/btc_whale_v2.json')
    print("✅ تم حفظ دماغ الحوت في: models/btc_whale_v2.json")

if __name__ == "__main__":
    train_brain_v2()