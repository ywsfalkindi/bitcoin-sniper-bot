import time
import os
from datetime import datetime
import sys

def main():
    print("==========================================")
    print("💎 BTC V7 SNIPER BOT - ACTIVATED")
    print("==========================================")
    print(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # حلقة لانهائية
    while True:
        try:
            # تشغيل ملف التوقع
            os.system('python 3_predict_v7.py')
            
            # الانتظار 5 دقائق (300 ثانية)
            # لماذا 5 دقائق؟ لأن النموذج مدرب على فريم الساعة، 
            # ولكن نريد التقاط الحركة بمجرد إغلاق الشمعة أو تحديث البيانات.
            print("⏳ Next scan in 5 minutes...\n")
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user.")
            sys.exit()
        except Exception as e:
            print(f"⚠️ Crash detected: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()