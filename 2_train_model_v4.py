import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import os

def train_brain_v4():
    print("🧠 (V4) تدريب 'القناص الذكي' مع إدارة المخاطر (ATR)...")
    
    if not os.path.exists('data/btc_data.csv'):
        print("❌ شغل 1_fetch_data_v4.py أولاً")
        return
    df = pd.read_csv('data/btc_data.csv')
    
    # --- 1. هندسة الميزات النسبية (Smart Features) ---
    # نستخدم النسب المئوية ليفهم النموذج السلوك وليس السعر
    df['returns'] = df['close'].pct_change()
    df['EMA_50'] = df.ta.ema(length=50)
    df['dist_EMA50'] = (df['close'] / df['EMA_50']) - 1
    df['RSI'] = df.ta.rsi(length=14)
    
    # أهم مؤشر: ATR النسبي (لقياس التذبذب)
    df['ATR'] = df.ta.atr(length=14)
    df['ATR_Pct'] = df['ATR'] / df['close']
    
    # كاشف الحيتان
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Whale_Activity'] = df['volume'] / df['Vol_MA20']
    
    # عرض البولنجر (للكشف عن الانفجارات السعرية)
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close']

    df.dropna(inplace=True)
    
    # --- 2. تحديد الهدف الذكي (Dynamic Target) ---
    # الهدف: هل سيرتفع السعر بمقدار (1.5 * ATR) خلال الـ 12 ساعة القادمة؟
    # هذا يعني أن الهدف يتغير حسب حالة السوق (في الهدوء هدف صغير، في الحركة القوية هدف كبير)
    
    FUTURE_PERIOD = 12
    MULTIPLIER = 1.5
    
    df['future_high'] = df['high'].rolling(window=FUTURE_PERIOD).max().shift(-FUTURE_PERIOD)
    
    # الشرط: هل أعلى سعر قادم أكبر من (سعر الإغلاق الحالي + 1.5 * ATR)؟
    df['Target'] = (df['future_high'] > (df['close'] + (df['ATR'] * MULTIPLIER))).astype(int)
    
    df.dropna(inplace=True)
    
    # --- 3. التجهيز والتدريب ---
    features = ['returns', 'dist_EMA50', 'RSI', 'ATR_Pct', 'Whale_Activity', 'BB_Width']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # إعدادات النموذج (مضبوطة لتقليل الإشارات الكاذبة)
    model = XGBClassifier(
        n_estimators=500,       # زيادة عدد الأشجار للتعلم العميق
        learning_rate=0.02,     # تعلم أبطأ لأدق النتائج
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=3      # التركيز على اقتناص الفرص النادرة (الشراء)
    )
    
    model.fit(X_train, y_train)
    
    # --- 4. التقييم ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds) # تهمنا الدقة في الشراء أكثر من أي شيء
    
    print(f"\n📊 دقة النموذج العام: {acc*100:.2f}%")
    print(f"🎯 دقة إشارات الشراء (Precision): {prec*100:.2f}% (هذا هو الرقم الأهم)")
    
    # حفظ النموذج
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_model('models/btc_v4_sniper.json')
    print("✅ تم حفظ النموذج V4 بنجاح.")

if __name__ == "__main__":
    train_brain_v4()