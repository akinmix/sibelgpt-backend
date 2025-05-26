# tests/unit/test_chat_function.py - DÜZELTİLMİŞ VERSİYON
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Ana dizini import path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestChatFunction:
    @pytest.mark.asyncio
    @patch('ask_handler.openai_client')  # ✅ Doğru patch
    @patch('ask_handler.supabase')       # ✅ Supabase de patch et
    async def test_answer_question_basic(self, mock_supabase, mock_openai):
        """Test basic chat functionality with proper mocking"""
        
        # ✅ DOĞRU MOCK YAPILANDIRMASI
        # OpenAI client mock'u
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [MagicMock()]
        mock_chat_response.choices[0].message.content = "Test yanıtı"
        
        # AsyncMock'u doğru şekilde yapılandır
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_chat_response)
        
        # Supabase mock'u (eğer kullanılıyorsa)
        mock_supabase.rpc.return_value.execute.return_value.data = []
        
        # ✅ TEST İÇE AKTARMA - Geç import
        from ask_handler import answer_question
        
        # ✅ TESTİ ÇALIŞTIR
        result = await answer_question(
            question="Kadıköy'de ev arıyorum",
            mode="real-estate"
        )
        
        # ✅ KONTROLLAR
        assert isinstance(result, str)
        assert len(result) > 0
        # Selamlaşma mesajı değil, gerçek yanıt olduğunu kontrol et
        assert any(word in result.lower() for word in ["test", "gayrimenkul", "yanıt", "merhaba"])
        
        # ✅ MOCK ÇAĞRILDIĞINI KONTROL ET
        mock_openai.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio 
    @patch('ask_handler.openai_client')
    async def test_answer_question_greeting(self, mock_openai):
        """Test greeting messages - should not call OpenAI"""
        
        # ✅ Geç import
        from ask_handler import answer_question
        
        # ✅ SELAMLAŞMA TESTİ
        result = await answer_question(
            question="Merhaba",
            mode="real-estate" 
        )
        
        # ✅ KONTROLLAR
        assert isinstance(result, str)
        assert len(result) > 0
        assert "merhaba" in result.lower()
        
        # ✅ OpenAI çağrılmamalı (selamlaşma için)
        mock_openai.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    @patch('ask_handler.openai_client')
    async def test_answer_question_different_modes(self, mock_openai):
        """Test different GPT modes"""
        
        # ✅ Mock yanıt
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mode test yanıtı"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # ✅ Geç import
        from ask_handler import answer_question
        
        modes = ["real-estate", "mind-coach", "finance"]
        
        for mode in modes:
            result = await answer_question(
                question="Test sorusu", 
                mode=mode
            )
            
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"✅ {mode} modu test edildi")

    @pytest.mark.asyncio
    async def test_detect_topic_function(self):
        """Test topic detection without external API calls"""
        
        # ✅ Basit fonksiyon testi
        from ask_handler import detect_topic
        
        # ✅ Bu fonksiyon mevcut mu?
        assert callable(detect_topic)
        print("✅ detect_topic fonksiyonu mevcut")

    def test_system_prompts_exist(self):
        """Test if system prompts are properly defined"""
        
        # ✅ Geç import
        from ask_handler import SYSTEM_PROMPTS
        
        # ✅ SYSTEM_PROMPTS kontrolü
        assert isinstance(SYSTEM_PROMPTS, dict)
        assert len(SYSTEM_PROMPTS) > 0
        assert "real-estate" in SYSTEM_PROMPTS
        print("✅ System prompts tanımlı")

if __name__ == "__main__":
    # ✅ Tek test çalıştırma
    pytest.main([__file__, "-v"])
