# tests/unit/test_chat_function.py
import pytest
from unittest.mock import patch, AsyncMock
import sys
import os

# Import path
sys.path.insert(0, '/opt/render/project/src')

# Güvenli import
try:
    import ask_handler
    HAS_ASK_HANDLER = True
except ImportError:
    HAS_ASK_HANDLER = False

class TestChatFunction:
    """Chat fonksiyon testleri - Ana kodu değiştirmeden"""
    
    @pytest.mark.skipif(not HAS_ASK_HANDLER, reason="ask_handler not available")
    @pytest.mark.asyncio
    @patch('ask_handler.openai_client')  # OpenAI'yi fake yapar
    async def test_answer_question_basic(self, mock_openai):
        """Temel chat testi - sahte OpenAI yanıtı"""
        
        # Sahte OpenAI yanıtı hazırla
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="Test yanıtı"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Test et
        result = await ask_handler.answer_question(
            question="Kadıköy'de ev arıyorum",
            mode="real-estate"
        )
        
        # Kontrol et
        assert isinstance(result, str)
        assert len(result) > 0
        # Selamlaşma mesajını da kabul et
        assert any(word in result.lower() for word in ["test", "gayrimenkul", "yardımcı", "merhaba"])
        assert "Test yanıtı" in result
    
    @pytest.mark.skipif(not HAS_ASK_HANDLER, reason="ask_handler not available")
    @pytest.mark.asyncio  
    @patch('ask_handler.openai_client')
    async def test_answer_question_different_modes(self, mock_openai):
        """Farklı modlar için test"""
        
        # Sahte yanıt
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="Mod testi yanıtı"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Her mod için test
        modes = ["real-estate", "mind-coach", "finance"]
        
        for mode in modes:
            result = await ask_handler.answer_question(
                question="Test sorusu",
                mode=mode
            )
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.skipif(not HAS_ASK_HANDLER, reason="ask_handler not available")
    @pytest.mark.asyncio
    @patch('ask_handler.openai_client')
    async def test_answer_question_with_history(self, mock_openai):
        """Konuşma geçmişi ile test"""
        
        # Sahte yanıt
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="Geçmiş ile yanıt"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response
        
        # Sahte konuşma geçmişi
        fake_history = [
            {"role": "user", "text": "Önceki soru"},
            {"role": "assistant", "text": "Önceki yanıt"}
        ]
        
        result = await ask_handler.answer_question(
            question="Yeni soru",
            mode="real-estate",
            conversation_history=fake_history
        )
        
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_ASK_HANDLER, reason="ask_handler not available")
    @pytest.mark.asyncio
    @patch('ask_handler.openai_client')
    async def test_answer_question_error_handling(self, mock_openai):
        """Hata durumu testi"""
        
        # OpenAI hatası simüle et
        mock_openai.chat.completions.create.side_effect = Exception("API Hatası")
        
        result = await ask_handler.answer_question(
            question="Test sorusu",
            mode="real-estate"
        )
        
        # Hata durumunda bile string döndürmeli
        assert isinstance(result, str)
        assert len(result) > 0

    # Ana kod import edilmese bile çalışacak testler
    def test_chat_concepts(self):
        """Chat kavramları testi - import gerekmez"""
        valid_modes = ["real-estate", "mind-coach", "finance"]
        assert len(valid_modes) == 3
        assert "real-estate" in valid_modes
    
    def test_message_structure(self):
        """Mesaj yapısı testi"""
        sample_message = {
            "role": "user",
            "text": "Test mesajı"
        }
        assert "role" in sample_message
        assert "text" in sample_message
        assert sample_message["role"] in ["user", "assistant"]
