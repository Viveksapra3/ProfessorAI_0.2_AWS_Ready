"""
Chat Service - Handles RAG-based conversations and multilingual support
"""

import time
from typing import Dict, Any
import config
from services.document_service import DocumentProcessor
from services.rag_service import RAGService
from services.llm_service import LLMService
from services.sarvam_service import SarvamService

class ChatService:
    """Main chat service that coordinates RAG, translation, and LLM services."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.sarvam_service = SarvamService()
        self.document_processor = DocumentProcessor()
        
        # Initialize RAGService, which will handle its own vectorstore creation
        self.rag_service = RAGService()
        self.vectorstore = self.rag_service.vectorstore
        
        if self.vectorstore:
            self.is_rag_active = True
            print("✅ RAG service initialized successfully.")
            # Now that vectorstore is confirmed, load any existing course content
            self._load_course_content_if_available()
        else:
            self.is_rag_active = False
            print("❗ RAG service could not initialize a vectorstore. Operating in general knowledge mode.")
    
    def _load_course_content_if_available(self):
        """Load generated course content into the vectorstore if available."""
        try:
            import os
            
            if os.path.exists(config.OUTPUT_JSON_PATH):
                print("📚 Found generated course content, loading into RAG system...")
                
                # Load course documents
                course_docs = self.document_processor.load_course_content_as_documents(config.OUTPUT_JSON_PATH)
                if course_docs:
                    # Split documents
                    print(f"📄 Loaded {len(course_docs)} course documents")
                   
                    split_course_docs = self.document_processor.split_documents(course_docs)
                    print(f"✂️ Split into {len(split_course_docs)} chunks")                    
                    # Create or update vectorstore
                    if self.vectorstore is None:
                        # Create new vectorstore with course content
                        self.vectorstore = self.document_processor.create_vectorstore_from_documents(split_course_docs)
                        print(f"✅ Created new vectorstore with {len(split_course_docs)} course content chunks")
                    else:
                        # Add to existing vectorstore
                        self.vectorstore.add_documents(split_course_docs)
                        print(f"✅ Added {len(split_course_docs)} course content chunks to existing vectorstore")
                    
                        
        except Exception as e:
            print(f"⚠️ Could not load course content: {e}")

    async def ask_question(self, query: str, query_language_code: str = "en-IN") -> Dict[str, Any]:
        """Answer a question using RAG with multilingual support."""
        
        response_lang_name = next(
            (lang["name"] for lang in config.SUPPORTED_LANGUAGES if lang["code"] == query_language_code), 
            "English"
        )

        if self.is_rag_active:
            # Translate query to English if needed
            start_time = time.time()
            english_query = query
            if query_language_code != "en-IN":
                print("[TASK] Translating query to English using Sarvam AI...")
                english_query = await self.sarvam_service.translate_text(
                    text=query,
                    source_language_code=query_language_code,
                    target_language_code="en-IN"
                )
                end_time = time.time()
                print(f"  > Translation complete in {end_time - start_time:.2f}s. (Query: '{english_query}')")
            
            try:
                # Execute RAG chain
                print("[TASK] Executing RAG chain...")
                start_time = time.time()
                answer = await self.rag_service.get_answer(english_query, response_lang_name)
                end_time = time.time()
                print(f"  > RAG chain complete in {end_time - start_time:.2f}s.")
                
                # Check if RAG found an answer
                if "I cannot find the answer" in answer:
                    print("  > RAG chain failed. Falling back to general LLM...")
                    start_time = time.time()
                    answer = await self.llm_service.get_general_response(query, response_lang_name)
                    end_time = time.time()
                    print(f"  > Fallback complete in {end_time - start_time:.2f}s.")
                    return {"answer": answer, "sources": ["General Knowledge Fallback"]}

                return {"answer": answer, "sources": ["Course Content"]}

            except Exception as e:
                print(f"  > Error during RAG chain invocation: {e}. Falling back...")
        
        # Fallback to general knowledge
        print("[TASK] Using general knowledge fallback...")
        start_time = time.time()
        answer = await self.llm_service.get_general_response(query, response_lang_name)
        end_time = time.time()
        print(f"  > General knowledge fallback complete in {end_time - start_time:.2f}s.")
        return {"answer": answer, "sources": ["General Knowledge"]}
    
    def update_with_course_content(self, course_data: dict):
        """Update the RAG system with new course content."""
        try:
            # Extract course documents
            course_documents = self.document_processor.extract_course_documents(course_data)
            
            if course_documents:
                # Split documents
                split_course_docs = self.document_processor.split_documents(course_documents)
                
                # Add to vectorstore
                if self.vectorstore:
                    self.vectorstore.add_documents(split_course_docs)
                else:
                    self.vectorstore = self.document_processor.create_vectorstore_from_documents(split_course_docs)
                    self.rag_service = RAGService(self.vectorstore)
                    self.is_rag_active = True
                
                print(f"✅ Added {len(split_course_docs)} course content chunks to RAG system")
                
        except Exception as e:
            print(f"⚠️ Error updating RAG with course content: {e}")
            raise e