"""
LLM Service - Handles OpenAI language model interactions with streaming support
"""

from openai import AsyncOpenAI
from typing import AsyncGenerator
import config

class LLMService:
    """Service for OpenAI LLM interactions."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=60.0  # 8 second timeout for all requests
        )
    
    async def get_general_response(self, query: str, target_language: str = "English") -> str:
        """Get a general response from the LLM."""
        messages = [
            {
                "role": "system", 
                "content": f"""You are a helpful AI assistant. Answer the user's question concisely and in {target_language}.
                
IMPORTANT QUALITY GUIDELINES:
- Provide clear, coherent, and well-structured responses
- Do NOT generate random characters, symbols, or gibberish
- Do NOT repeat the same content unnecessarily
- Do NOT provide incomplete or truncated responses
- Always respond in proper, grammatically correct {target_language}
- Ensure your response is relevant and directly addresses the question
- Be helpful, informative, and professional"""
            },
            {"role": "user", "content": query}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME, 
                messages=messages, 
                temperature=1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting general LLM response: {e}")
            return "I am sorry, I couldn't process that request at the moment."
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate text using the LLM."""
        if target_language.lower() == "english":
            return text
            
        messages = [
            {
                "role": "system", 
                "content": f"""You are an expert translation assistant. Translate the following text into {target_language}. Respond with only the translated text.
                
QUALITY REQUIREMENTS:
- Provide accurate, natural-sounding translation
- Do NOT add random characters or symbols
- Maintain the original meaning and tone
- Use proper grammar and syntax in {target_language}"""
            },
            {"role": "user", "content": text}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME, 
                messages=messages, 
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during LLM translation: {e}")
            return text
    
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response from the LLM."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant.
                
QUALITY REQUIREMENTS:
- Provide clear, coherent responses
- Do NOT generate random characters or gibberish
- Ensure responses are complete and well-formed
- Use proper grammar and formatting"""
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "I apologize, but I couldn't generate a response at the moment."
    
    async def generate_response_stream(self, prompt: str, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """Stream response generation from the LLM."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant.
                
QUALITY REQUIREMENTS:
- Provide clear, coherent responses
- Do NOT generate random characters or gibberish
- Ensure responses are complete and well-formed
- Use proper grammar and formatting"""
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            stream = await self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error in streaming LLM response: {e}")
            yield "I apologize, but I couldn't generate a response at the moment."