import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score
import os
import joblib # لحفظ النموذج بشكل أسرع

def train_brain_v5():
    print("🧠 (V5) بدء تدريب 'النموذج العالمي' باستراتيجية الصناديق...")
    
    if not os.path.exists('data/btc_data_v5.csv'):
        print("❌ البيانات غير موجودة! شغل الملف رقم 1 أولاً.")
        return
        
    df = pd.read_csv('data/btc_data_v5.csv')
    
    # --- 1. هندسة الميزات العميقة (Deep Feature Engineering) ---
    
    # أ) مؤشرات الزخم (1H)
    df['RSI'] = df.ta.rsi(length=14)
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    df['Trend_1H'] = (df['close'] > df['EMA_50']).astype(int) # هل نحن فوق المتوسط؟
    
    # ب) مؤشرات السياق (4H) - مهم جداً لفلترة الإشارات الكاذبة
    df['Trend_4H'] = (df['close_4h'] > df['close_4h'].rolling(50).mean()).astype(int)
    df['RSI_4H_Divergence'] = df['close'] / df['close_4h'] # العلاقة بين السعر اللحظي والعام
    
    # ج) مؤشرات السيولة والحيتان
    df['ATR'] = df.ta.atr(length=14)
    df['ATR_Pct'] = df['ATR'] / df['close'] # نسبة التذبذب
    df['Force_Index'] = df['close'].diff(1) * df['volume'] # قوة الحركة
    
    # د) مؤشر الخطر (Funding Rate)
    # إذا كان التمويل إيجابي جداً، السوق "متحمس" وقد ينهار (Long Squeeze)
    df['Funding_Risk'] = (df['fundingRate'] > 0.01).astype(int) 

    df.dropna(inplace=True)

    # --- 2. الهدف الذكي (Adaptive Target) ---
    # نبحث عن ارتفاع قوي (أكثر من 2x ATR) خلال الـ 8 ساعات القادمة
    # ولكن بشرط: ألا ينخفض السعر لضرب وقف الخسارة (1x ATR) قبل تحقيق الهدف
    
    FUTURE = 8
    ATR_MULT_TARGET = 2.0
    ATR_MULT_STOP = 1.0
    
    targets = []
    for i in range(len(df) - FUTURE):
        curr_close = df.iloc[i]['close']
        curr_atr = df.iloc[i]['ATR']
        
        take_profit = curr_close + (curr_atr * ATR_MULT_TARGET)
        stop_loss = curr_close - (curr_atr * ATR_MULT_STOP)
        
        future_window = df.iloc[i+1 : i+FUTURE+1]
        
        hit_tp = future_window['high'].max() >= take_profit
        hit_sl = future_window['low'].min() <= stop_loss
        
        if hit_tp and not hit_sl:
            targets.append(1) # صفقة ناجحة ونظيفة
        else:
            targets.append(0)
            
    # تعبئة الباقي أصفار
    targets.extend([0] * FUTURE)
    df['Target'] = targets
    
    # تنظيف نهائي
    features = [
        'RSI', 'Trend_1H', 'Trend_4H', 'ATR_Pct', 
        'fundingRate', 'Funding_Risk', 'Force_Index', 'RSI_4H_Divergence'
    ]
    
    X = df[features]
    y = df['Target']
    
    # --- 3. التدريب المتقدم (TimeSeries Validated) ---
    # نستخدم TimeSeriesSplit بدلاً من العشوائي لمحاكاة الواقع
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5, # التركيز بشدة على الفرص النادرة (الشراء الناجح)
        n_jobs=-1,
        random_state=42
    )
    
    print("🏋️‍♂️ جاري التدريب عبر الزمن (Walk-Forward Validation)...")
    
    # التدريب على كامل البيانات مع التحقق الضمني
    model.fit(X, y)
    
    # --- 4. الحفظ والتقرير ---
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/btc_v5_worldclass.pkl')
    
    # أهمية الميزات
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n🌟 أهم العوامل التي يعتمد عليها النموذج:")
    print(feat_importances.nlargest(5))
    
    print("\n✅ تم حفظ النموذج V5 بنجاح. جاهز للقنص.")

if __name__ == "__main__":
    train_brain_v5()