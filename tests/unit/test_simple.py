# run_simple_test.py - DOĞRUDAN TEST ÇALIŞTIRICI
import sys
import os
import asyncio
from pathlib import Path

# Path ayarlama
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 Basit Test Çalıştırıcı")
print(f"📁 Proje dizini: {project_root}")

async def run_basic_test():
    """Basit chat test"""
    try:
        # Test import
        print("📦 ask_handler import ediliyor...")
        from ask_handler import answer_question
        print("✅ ask_handler başarıyla import edildi")
        
        # Basit test
        print("🧪 Basit selamlaşma testi...")
        result = await answer_question("Merhaba", "real-estate")
        
        print(f"✅ TEST SONUCU:")
        print(f"📝 Yanıt uzunluğu: {len(result)} karakter")
        print(f"📝 İlk 100 karakter: {result[:100]}...")
        
        # Test kontrolleri
        assert isinstance(result, str), "Sonuç string olmalı"
        assert len(result) > 0, "Boş yanıt olmamalı"
        assert "merhaba" in result.lower(), "Selamlaşma yanıtı olmalı"
        
        print("🎉 TÜM TESTLER BAŞARILI!")
        return True
        
    except Exception as e:
        print(f"❌ TEST HATASI: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_basic_test())
    if success:
        print("✅ TEST BAŞARILI - SİSTEM ÇALIŞIYOR!")
    else:
        print("❌ TEST BAŞARISIZ")
        sys.exit(1)
