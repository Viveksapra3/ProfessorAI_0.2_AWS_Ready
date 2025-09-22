"""
Audio Service - Handles audio transcription and generation
"""

import io
from typing import Optional
import config
from services.sarvam_service import SarvamService
from utils.connection_monitor import is_client_connected, is_normal_closure

class AudioService:
    """Service for audio processing operations."""
    
    def __init__(self):
        self.sarvam_service = SarvamService()
    
    async def transcribe_audio(self, audio_file_buffer: io.BytesIO, language: Optional[str] = None) -> str:
        """Transcribe audio to text."""
        effective_language = language or config.SUPPORTED_LANGUAGES[0]['code']
        return await self.sarvam_service.transcribe_audio(audio_file_buffer, effective_language)
    
    async def generate_audio_from_text(self, text: str, language: Optional[str] = None, ultra_fast: bool = False) -> io.BytesIO:
        """Generate audio from text with speed options."""
        effective_language = language or config.SUPPORTED_LANGUAGES[0]['code']
        
        if ultra_fast:
            return await self.sarvam_service.generate_audio_ultra_fast(
                text, 
                effective_language, 
                config.SARVAM_TTS_SPEAKER
            )
        else:
            return await self.sarvam_service.generate_audio(
                text, 
                effective_language, 
                config.SARVAM_TTS_SPEAKER
            )
    
    async def stream_audio_from_text(self, text: str, language: Optional[str] = None, websocket=None):
        """Stream audio chunks as they're generated for real-time playback."""
        effective_language = language or config.SUPPORTED_LANGUAGES[0]['code']
        
        try:
            async for audio_chunk in self.sarvam_service.stream_audio_generation(
                text, 
                effective_language, 
                config.SARVAM_TTS_SPEAKER,
                websocket
            ):
                if audio_chunk and len(audio_chunk) > 0:
                    yield audio_chunk
        except Exception as e:
            error_msg = str(e)
            # Check if this is a normal disconnection
            if self._is_normal_disconnection(error_msg):
                print(f"🔌 Client disconnected during audio streaming: {e}")
                print(f"⚠️ Stopping audio streaming - client no longer connected")
                return
            else:
                print(f"❌ Error in audio streaming: {e}")
                # Only fallback for actual errors, not disconnections
                if not websocket or not self._is_client_disconnected(websocket):
                    try:
                        audio_buffer = await self.generate_audio_from_text(text, language, ultra_fast=True)
                        if audio_buffer and audio_buffer.getbuffer().nbytes > 0:
                            # Yield the entire audio as a single chunk
                            yield audio_buffer.getvalue()
                    except Exception as fallback_error:
                        print(f"Fallback audio generation also failed: {fallback_error}")
                        # Return empty generator
                        return
    
    def _is_client_disconnected(self, websocket) -> bool:
        """Check if WebSocket client is disconnected."""
        try:
            if not websocket:
                return False
            
            # Check if WebSocket is closed or closing
            if hasattr(websocket, 'closed') and websocket.closed:
                return True
            
            if hasattr(websocket, 'state'):
                # WebSocket states: CONNECTING=0, OPEN=1, CLOSING=2, CLOSED=3
                return websocket.state in [2, 3]  # CLOSING or CLOSED
            
            return False
        except Exception:
            # If we can't check the state, assume disconnected for safety
            return True
    
    def _is_normal_disconnection(self, error_msg: str) -> bool:
        """Check if error message indicates a normal client disconnection."""
        if not error_msg:
            return False
        
        error_msg = str(error_msg).lower()
        
        # Check for normal WebSocket closure codes
        normal_codes = ["1000", "1001"]  # OK, Going Away
        for code in normal_codes:
            if code in error_msg:
                return True
        
        # Check for common disconnection phrases
        disconnection_phrases = [
            "connection closed",
            "client disconnected", 
            "going away",
            "connection lost"
        ]
        
        for phrase in disconnection_phrases:
            if phrase in error_msg:
                return True
        
        return False