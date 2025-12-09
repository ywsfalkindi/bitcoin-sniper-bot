import time
import os
from datetime import datetime

def run_bot_v5():
    print("🛡️ تشغيل النظام V5 (World Class AI Trader)...")
    print("جاري المزامنة مع الأسواق العالمية...")
    
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"⏰ وقت الفحص: {datetime.now().strftime('%H:%M:%S')}")
            
            # تشغيل المحلل
            os.system('python 3_predict_v5.py')
            
            # الانتظار الذكي: 15 دقيقة
            print("\n⏳ النظام في وضع الاستعداد للدورة القادمة (15 دقيقة)...")
            time.sleep(900) 
            
        except KeyboardInterrupt:
            print("\n🛑 تم إيقاف النظام يدوياً.")
            break
        except Exception as e:
            print(f"⚠️ خطأ في الحلقة: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot_v5()