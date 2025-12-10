import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

# ==========================================
# 1. هندسة الميزات (V9.1 - المنقحة)
# ==========================================
def feature_engineering_v9(df):
    data = df.copy()
    
    # 1. ضبط الفهرس
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            data.sort_index(inplace=True)
    
    # 2. الحسابات الرياضية
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    
    # Garman-Klass Volatility
    data['GK_Vol'] = ((np.log(data['high'] / data['low'])**2) / 2) - \
                     (2 * np.log(2) - 1) * ((np.log(data['close'] / data['open'])**2))
    
    # 3. ميزات الزخم والسيولة
    data['RSI'] = data.ta.rsi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14']
    data['ATR'] = data.ta.atr(length=14)
    
    # VWAP مع حماية
    data['vwap'] = data.ta.vwap()
    if data['vwap'].isnull().all():
         data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['dist_vwap'] = (data['close'] - data['vwap']) / (data['vwap'] + 1e-9)
    
    # 4. فلتر التذبذب (جديد): هل السوق يتحرك أصلاً؟
    # سنستخدم هذا لاحقاً لتجاهل التذبذب الميت
    data['Is_Choppy'] = np.where((data['ADX'] < 20) & (data['RSI'].between(40, 60)), 1, 0)

    data.dropna(inplace=True)
    return data

# ==========================================
# 2. تحديد الأهداف (مخفف وأكثر واقعية)
# ==========================================
def labeling_triple_barrier(df, horizon=24, vol_mult=1.0):
    # زدنا الأفق الزمني لـ 24 ساعة لكي نعطي الصفقة نفساً
    # خفضنا مضاعف التقلب لـ 1.0 لنجعل الأهداف أقرب
    targets = []
    prices = df['close'].values
    atr = df['ATR'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(len(df) - horizon):
        curr = prices[i]
        limit = atr[i] * vol_mult
        
        # الهدف: 1.2 ضعف المخاطرة (أكثر واقعية من 1.8)
        tp = curr + (limit * 1.2) 
        sl = curr - (limit * 1.0)
        
        outcome = 0
        for j in range(1, horizon + 1):
            if i+j >= len(df): break
            h = highs[i+j]
            l = lows[i+j]
            
            if l <= sl:
                outcome = 0 # ضرب الوقف
                break
            if h >= tp:
                outcome = 1 # ضرب الهدف
                break
        targets.append(outcome)
    return targets

# ==========================================
# 3. التدريب الذكي (Auto-Tuning)
# ==========================================
def train_brain_v9():
    print("🧠 (V9.1 Stabilized) بدء التدريب مع التصحيح الذاتي...")
    
    if not os.path.exists('data/btc_data_v9.csv'):
        print("❌ البيانات مفقودة!")
        return

    df = pd.read_csv('data/btc_data_v9.csv')
    df = feature_engineering_v9(df)
    
    print("🎯 تحديد الأهداف (Easy Mode)...")
    targets = labeling_triple_barrier(df)
    
    df = df.iloc[:len(targets)].copy()
    df['Target'] = targets
    
    # حذف أوقات "موت السوق" من التدريب (لتقليل الضجيج)
    print(f"📉 قبل التنظيف: {len(df)} سجل")
    df = df[df['Is_Choppy'] == 0]
    print(f"📈 بعد إزالة التذبذب الميت: {len(df)} سجل")
    
    features = ['log_ret', 'GK_Vol', 'dist_vwap', 'RSI', 'ADX', 'ATR']
    if 'fng_value' in df.columns: features.append('fng_value')
    
    X = df[features]
    y = df['Target']
    
    # طباعة توازن البيانات
    win_rate_base = (y.sum() / len(y)) * 100
    print(f"📊 نسبة الصفقات الناجحة في البيانات: {win_rate_base:.2f}%")
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    best_thresholds = []

    print("\n🔬 التحقق الزمني والبحث عن العتبة (Threshold Search):")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # موازنة
        if len(y_train.unique()) < 2: continue
        ratio = np.sqrt((y_train==0).sum() / (y_train==1).sum())
        
        # نماذج أخف وأسرع (Less Overfitting)
        clf1 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, scale_pos_weight=ratio, n_jobs=-1, random_state=42)
        clf2 = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, auto_class_weights='SqrtBalanced', verbose=False, random_state=42)
        clf3 = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, class_weight='balanced', verbose=-1, random_state=42)
        
        ensemble = VotingClassifier(estimators=[('xgb', clf1), ('cat', clf2), ('lgbm', clf3)], voting='soft')
        ensemble.fit(X_train, y_train)
        
        # اختبار العتبات
        probs = ensemble.predict_proba(X_test)[:, 1]
        
        fold_best_prec = 0
        fold_best_thresh = 0.5
        fold_trades = 0
        
        # نجرب عتبات مختلفة لكل فترة زمنية
        for t in np.arange(0.5, 0.95, 0.05):
            preds = (probs >= t).astype(int)
            if preds.sum() > 5: # شرط وجود 5 صفقات على الأقل
                p = precision_score(y_test, preds, zero_division=0)
                if p > fold_best_prec:
                    fold_best_prec = p
                    fold_best_thresh = t
                    fold_trades = preds.sum()
        
        print(f"Fold {fold+1}: Best Thresh={fold_best_thresh:.2f} | Precision={fold_best_prec:.2%} | Trades={fold_trades}")
        if fold_trades > 0:
            scores.append(fold_best_prec)
            best_thresholds.append(fold_best_thresh)

    avg_prec = np.mean(scores) if scores else 0
    avg_thresh = np.mean(best_thresholds) if best_thresholds else 0.70
    
    print(f"\n🏆 متوسط الدقة المتوقعة: {avg_prec:.2%}")
    print(f"🔑 العتبة الموصى بها: {avg_thresh:.2f}")

    # التدريب النهائي
    print("🚀 تدريب النموذج النهائي...")
    final_ratio = np.sqrt((y==0).sum() / (y==1).sum())
    final_model = VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, scale_pos_weight=final_ratio, n_jobs=-1)),
            ('cat', CatBoostClassifier(iterations=500, depth=5, learning_rate=0.02, auto_class_weights='SqrtBalanced', verbose=False)),
            ('lgbm', LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, class_weight='balanced', verbose=-1))
        ], voting='soft'
    )
    final_model.fit(X, y)
    
    # حفظ العتبة مع النموذج
    model_data = {'model': final_model, 'threshold': avg_thresh}
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model_data, 'models/btc_v9_worldclass.pkl')
    print("✅ تم الحفظ (النموذج + العتبة المثالية).")

if __name__ == "__main__":
    train_brain_v9()