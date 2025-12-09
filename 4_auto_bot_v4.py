import time
import os
from datetime import datetime

def run_bot():
    print("🤖 تم تشغيل (Sniper Bot V4)...")
    print("سيقوم البوت بمسح السوق كل 15 دقيقة بحثاً عن الفرص الذهبية.")
    
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"⏰ فحص جديد: {datetime.now().strftime('%H:%M:%S')}")
            
            # تشغيل المحلل الذكي V4
            os.system('python 3_predict_v4.py')
            
            print("\n⏳ الانتظار للدورة القادمة (15 دقيقة)...")
            time.sleep(900) 
            
        except KeyboardInterrupt:
            print("\n🛑 تم الإيقاف.")
            break
        except Exception as e:
            print(f"⚠️ خطأ: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot()