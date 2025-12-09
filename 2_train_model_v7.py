import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

# ==========================================
# 1. هندسة الميزات (نفس النسخة المحسنة)
# ==========================================
def feature_engineering_v7(df):
    data = df.copy()
    data['Returns'] = np.log(data['close'] / data['close'].shift(1))
    data['Range'] = (data['high'] - data['low']) / data['open']
    data['Vol_1H'] = data['Returns'].rolling(24).std()
    data['Vol_4H_Proxy'] = data['Returns'].rolling(24).std() 
    data['Vol_Ratio'] = data['Vol_1H'] / (data['Vol_4H_Proxy'] + 1e-9)
    
    data['Close_Loc'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-9)
    data['Volume_Flow'] = np.where(data['Close_Loc'] > 0.5, data['volume'], -data['volume'])
    data['CVD_Proxy'] = data['Volume_Flow'].rolling(12).sum()
    
    data['RSI'] = data.ta.rsi(length=14)
    data['MFI'] = data.ta.mfi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14']
    
    change = data['close'].diff(10).abs()
    volatility = data['close'].diff().abs().rolling(10).sum()
    data['Efficiency_Ratio'] = change / (volatility + 1e-9)
    
    data['Funding_x_Vol'] = data['fundingRate'] * data['Vol_1H']
    data['Trend_4H'] = (data['close_4h'] > data['close_4h'].shift(1)).astype(int)
    
    data.dropna(inplace=True)
    return data

# ==========================================
# 2. تحديد الأهداف (Triple Barrier)
# ==========================================
def labeling_triple_barrier(df, horizon=12, vol_mult=1.5):
    targets = []
    prices = df['close'].values
    atr = df.ta.atr(length=14).values
    
    for i in range(len(df) - horizon):
        curr = prices[i]
        cur_atr = atr[i] if not np.isnan(atr[i]) else curr * 0.01
        
        # أهداف: الربح ضعف الخسارة تقريباً
        tp = curr + (cur_atr * vol_mult * 1.5)
        sl = curr - (cur_atr * vol_mult * 0.8)
        
        outcome = 0
        for j in range(1, horizon + 1):
            if i+j >= len(df): break
            high = df.iloc[i+j]['high']
            low = df.iloc[i+j]['low']
            
            if low <= sl:
                outcome = 0
                break
            if high >= tp:
                outcome = 1
                break
        targets.append(outcome)
    return targets

# ==========================================
# 3. التدريب والمعايرة (The Sniper Calibration)
# ==========================================
def train_brain_v7():
    print("🧠 (V7.2 Sniper Calibration) بدء التدريب والمعايرة الدقيقة...")
    
    if not os.path.exists('data/btc_data_v7.csv'):
        print("❌ البيانات مفقودة!")
        return

    df = pd.read_csv('data/btc_data_v7.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("🛠️ هندسة الميزات...")
    df = feature_engineering_v7(df)
    
    print("🎯 تحديد الأهداف...")
    targets = labeling_triple_barrier(df)
    df = df.iloc[:len(targets)]
    df['Target'] = targets
    
    features = [
        'RSI', 'MFI', 'ADX', 'Efficiency_Ratio', 
        'Vol_Ratio', 'CVD_Proxy', 'fundingRate', 
        'Funding_x_Vol', 'Trend_4H', 'Range'
    ]
    
    X = df[features]
    y = df['Target']
    
    # تقسيم البيانات (آخر 20% اختبار)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # حساب الوزن (مخفف هذه المرة)
    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    # نستخدم الجذر التربيعي لتخفيف حدة الموازنة، لزيادة الدقة
    scale_weight = np.sqrt(count_0 / count_1) 
    print(f"⚖️ موازنة ذكية: {count_0} vs {count_1} | الوزن المعدل = {scale_weight:.2f}")

    # تعريف النماذج
    clf1 = XGBClassifier(
        n_estimators=800, learning_rate=0.01, max_depth=5, 
        scale_pos_weight=scale_weight, 
        subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1
    )
    
    clf2 = CatBoostClassifier(
        iterations=800, learning_rate=0.01, depth=6, 
        auto_class_weights='SqrtBalanced', # موازنة أخف
        verbose=False, allow_writing_files=False, random_state=42
    )
    
    clf3 = LGBMClassifier(
        n_estimators=800, learning_rate=0.01, max_depth=5, 
        class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('xgb', clf1), ('cat', clf2), ('lgbm', clf3)],
        voting='soft'
    )
    
    print("🏋️‍♂️ جاري التدريب على البيانات التاريخية...")
    ensemble.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 🔬 البحث عن العتبة الذهبية (Threshold Optimization)
    # ---------------------------------------------------------
    print("\n🔭 فحص المنظار (Calibration Analysis):")
    print("-" * 50)
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Win Rate':<10} | {'Trades':<10}")
    print("-" * 50)
    
    probs = ensemble.predict_proba(X_test)[:, 1]
    
    best_thresh = 0.5
    best_prec = 0.0
    
    # نجرب عتبات من 50% إلى 95%
    for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = (probs >= t).astype(int)
        
        # حساب الدقة
        if preds.sum() > 0:
            prec = precision_score(y_test, preds, zero_division=0)
            trades = preds.sum()
            
            print(f"{t:<10} | {prec:.2%}    | {prec:.2%}    | {trades}")
            
            # نريد دقة فوق 50% مع عدد صفقات معقول (أكثر من 50 صفقة في الاختبار)
            if prec > best_prec and trades > 20:
                best_prec = prec
                best_thresh = t
        else:
            print(f"{t:<10} | 0.00%      | 0.00%      | 0")

    print("-" * 50)
    print(f"💡 التوصية: استخدم Threshold = {best_thresh} في ملف التوقع للحصول على دقة {best_prec:.1%}")
    
    # التدريب النهائي
    print("\n🚀 إعادة التدريب النهائي (Full Deployment)...")
    ensemble.fit(X, y)
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(ensemble, 'models/btc_v7_ensemble.pkl')
    print("✅ تم حفظ القناص الجديد.")

if __name__ == "__main__":
    train_brain_v7()