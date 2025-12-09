import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def feature_engineering(df):
    """ هندسة الميزات المتقدمة (The Secret Sauce) """
    data = df.copy()
    
    # 1. Heikin Ashi (لتقليل الضوضاء)
    data['HA_Close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    data['HA_Open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2
    data['HA_High'] = data[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    data['HA_Low'] = data[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    
    # 2. مؤشرات الزخم المتقدمة
    data['RSI'] = data.ta.rsi(length=14)
    data['ADX'] = data.ta.adx(length=14)['ADX_14'] 
    
    # 3. Volatility (Garman-Klass)
    data['Log_Ret'] = np.log(data['close'] / data['close'].shift(1))
    data['Volatility'] = data['Log_Ret'].rolling(window=24).std() * np.sqrt(24)
    data['ATR'] = data.ta.atr(length=14)
    
    # 4. Z-Score
    data['Z_Score'] = (data['close'] - data['close'].rolling(20).mean()) / (data['close'].rolling(20).std() + 1e-9)
    
    # 5. Cyclical Time Features
    data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
    
    # 6. Proxy Order Flow
    data['Buying_Pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-9) * data['volume']
    
    # 7. Context
    data['Trend_4H'] = (data['close_4h'] > data['close_4h'].shift(1)).astype(int)
    data['Divergence'] = data['close'] / data['close_4h']
    
    data.dropna(inplace=True)
    return data

def labeling_triple_barrier(df, atr_mult_tp=2.5, atr_mult_sl=1.0, horizon=12):
    """ نظام الأهداف الثلاثي """
    targets = []
    for i in range(len(df) - horizon):
        curr_close = df.iloc[i]['close']
        curr_atr = df.iloc[i]['ATR']
        
        tp = curr_close + (curr_atr * atr_mult_tp)
        sl = curr_close - (curr_atr * atr_mult_sl)
        
        future = df.iloc[i+1 : i+horizon+1]
        
        hit_tp = False
        hit_sl = False
        
        for _, row in future.iterrows():
            if row['low'] <= sl:
                hit_sl = True
                break
            if row['high'] >= tp:
                hit_tp = True
                break
                
        if hit_tp and not hit_sl:
            targets.append(1)
        else:
            targets.append(0)
            
    return targets

def train_brain_v6():
    print("🧠 (V6) بدء تدريب النموذج العالمي...")
    
    if not os.path.exists('data/btc_data_v6.csv'):
        print("❌ البيانات مفقودة!")
        return

    df = pd.read_csv('data/btc_data_v6.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. التجهيز
    print("🛠️ هندسة الميزات...")
    df = feature_engineering(df)
    
    print("🎯 تحديد الأهداف...")
    targets = labeling_triple_barrier(df)
    
    df = df.iloc[:len(targets)]
    df['Target'] = targets
    
    features = [
        'RSI', 'ADX', 'Z_Score', 'Volatility', 
        'Buying_Pressure', 'fundingRate', 
        'hour_sin', 'hour_cos', 'Trend_4H', 'Divergence'
    ]
    
    X = df[features]
    y = df['Target']
    
    if len(X) < 100:
        print("⚠️ البيانات قليلة جداً.")
        return

    # حساب الوزن
    neg, pos = np.bincount(y)
    scale = neg / pos if pos > 0 else 1
    
    # إعدادات النموذج المشتركة
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.005,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'scale_pos_weight': scale,
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'random_state': 42
    }

    # --- المرحلة 1: التحقق (Cross Validation) مع الإيقاف المبكر ---
    print("🏋️‍♂️ التدريب والتحقق (Cross-Validation)...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    try:
        # هنا نستخدم نسخة مع early_stopping_rounds
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # نعرف نموذجاً مؤقتاً للاختبار
            cv_model = XGBClassifier(**model_params, early_stopping_rounds=50)
            cv_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        print("   ✅ انتهى التحقق بنجاح.")

        # --- المرحلة 2: التدريب النهائي (Production) بدون إيقاف مبكر ---
        print("🚀 جاري بناء النموذج النهائي على كامل البيانات...")
        
        # نعرف النموذج النهائي (بدون early_stopping_rounds في المُنشئ)
        final_model = XGBClassifier(**model_params)
        
        # ندربه على كل البيانات
        final_model.fit(X, y, verbose=False)
        
        if not os.path.exists('models'): os.makedirs('models')
        joblib.dump(final_model, 'models/btc_v6_model.pkl')
        
        print("\n📊 أهم العوامل المؤثرة:")
        imps = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
        print(imps.head(6))
        print("\n✅ تم بناء النموذج V6 بنجاح وجاهز للقنص!")
        
    except Exception as e:
        print(f"❌ حدث خطأ أثناء التدريب: {e}")

if __name__ == "__main__":
    train_brain_v6()