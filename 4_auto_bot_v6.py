import time
import os
from datetime import datetime

def main_loop():
    print("=========================================")
    print("🤖 BTC AI TRADER V6 (World Class Edition)")
    print("=========================================")
    
    # تأكد من وجود ملفات البيانات والنماذج
    if not os.path.exists('data'): os.makedirs('data')
    if not os.path.exists('models'): os.makedirs('models')
    
    print("1️⃣ التحقق من تحديث البيانات...")
    # يمكن تشغيل ملف الجلب مرة واحدة عند البدء أو كل فترة طويلة
    # os.system('python 1_fetch_data_v6.py') 
    
    print("2️⃣ بدء حلقة المراقبة...")
    while True:
        try:
            now = datetime.now().strftime('%H:%M:%S')
            print(f"\n⏰ فحص السوق: {now}")
            
            # تشغيل القناص
            os.system('python 3_predict_v6.py')
            
            # انتظر 15 دقيقة (900 ثانية) - أو حسب استراتيجيتك
            # للمضاربة السريعة جداً يمكن جعلها 5 دقائق، لكن النموذج تدرب على 1H
            print("⏳ استراحة المحارب (15 دقيقة)...")
            time.sleep(900)
            
        except KeyboardInterrupt:
            print("\n🛑 إيقاف النظام.")
            break
        except Exception as e:
            print(f"⚠️ خطأ غير متوقع: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()