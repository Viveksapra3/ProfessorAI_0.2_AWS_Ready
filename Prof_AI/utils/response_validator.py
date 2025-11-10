"""
Response Validator - Validates LLM responses to prevent garbage output
"""

import re
import logging

class ResponseValidator:
    """Validates LLM responses for quality and coherence."""
    
    # Patterns that indicate garbage or low-quality responses
    GARBAGE_PATTERNS = [
        r'^[\W_]{10,}$',  # Only special characters
        r'(.)\1{20,}',     # Same character repeated 20+ times
        r'^[^a-zA-Z\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0600-\u06FF]{50,}$',  # No readable characters
    ]
    
    @staticmethod
    def is_valid_response(response: str, min_length: int = 5, max_repetition: int = 5) -> bool:
        """
        Check if a response is valid and not garbage.
        
        Args:
            response: The response text to validate
            min_length: Minimum acceptable response length
            max_repetition: Maximum allowed word repetitions
            
        Returns:
            bool: True if response is valid, False otherwise
        """
        if not response or not isinstance(response, str):
            logging.warning("Invalid response: empty or not a string")
            return False
        
        # Check minimum length
        if len(response.strip()) < min_length:
            logging.warning(f"Response too short: {len(response.strip())} chars")
            return False
        
        # Check for garbage patterns
        for pattern in ResponseValidator.GARBAGE_PATTERNS:
            if re.search(pattern, response):
                logging.warning(f"Garbage pattern detected: {pattern}")
                return False
        
        # Check for excessive word repetition
        words = response.split()
        if len(words) > 10:  # Only check if response has enough words
            word_counts = {}
            for word in words:
                word_lower = word.lower().strip('.,!?;:')
                if len(word_lower) > 3:  # Only count meaningful words
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            # Check if any word appears too many times
            for word, count in word_counts.items():
                if count > max_repetition and count > len(words) * 0.3:
                    logging.warning(f"Excessive repetition of word '{word}': {count} times")
                    return False
        
        # Check for reasonable character distribution
        alpha_count = sum(1 for c in response if c.isalpha())
        total_chars = len(response)
        
        if total_chars > 20 and alpha_count / total_chars < 0.3:
            logging.warning(f"Too few alphabetic characters: {alpha_count}/{total_chars}")
            return False
        
        return True
    
    @staticmethod
    def sanitize_response(response: str) -> str:
        """
        Clean up and sanitize a response.
        
        Args:
            response: The response to sanitize
            
        Returns:
            str: Sanitized response
        """
        if not response:
            return ""
        
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response)
        
        # Remove excessive punctuation repetition
        response = re.sub(r'([!?.]){4,}', r'\1\1\1', response)
        
        # Trim
        response = response.strip()
        
        return response
    
    @staticmethod
    def validate_and_sanitize(response: str, fallback_message: str = None) -> str:
        """
        Validate and sanitize a response, returning fallback if invalid.
        
        Args:
            response: The response to validate
            fallback_message: Message to return if validation fails
            
        Returns:
            str: Valid response or fallback message
        """
        sanitized = ResponseValidator.sanitize_response(response)
        
        if ResponseValidator.is_valid_response(sanitized):
            return sanitized
        
        logging.error(f"Response validation failed for: {response[:100]}...")
        
        if fallback_message:
            return fallback_message
        
        return "I apologize, but I encountered an issue generating a proper response. Please try rephrasing your question."
