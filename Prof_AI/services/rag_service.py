"""
RAG Service - Handles Retrieval-Augmented Generation
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from typing import List, Any
import config

class RAGService:
    """Service for RAG-based question answering."""
    
    def __init__(self, vectorstore: Chroma = None):
        from langchain_openai import OpenAIEmbeddings

        if vectorstore is None:
            if config.USE_CHROMA_CLOUD:
                from core.cloud_vectorizer import CloudVectorizer
                cloud_vectorizer = CloudVectorizer()
                self.vectorstore = cloud_vectorizer.get_vector_store()
            else:
                from core.vectorizer import Vectorizer
                embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME, openai_api_key=config.OPENAI_API_KEY, chunk_size=200)
                self.vectorstore = Vectorizer.load_vector_store(config.FAISS_DB_PATH, embeddings)
                if not self.vectorstore:
                    # If loading fails, create an empty FAISS store to avoid crashing
                    from langchain_community.vectorstores import FAISS
                    import numpy as np
                    # Create a dummy document and embedding to initialize an empty store
                    dummy_doc = ["Initial empty document"]
                    dummy_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME, openai_api_key=config.OPENAI_API_KEY, chunk_size=200)
                    self.vectorstore = FAISS.from_texts(dummy_doc, dummy_embeddings)
        else:
            self.vectorstore = vectorstore

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=config.GROQ_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_template(config.QA_PROMPT_TEMPLATE)
        self.retriever = self.vectorstore.as_retriever(
            search_type=config.RETRIEVAL_SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVAL_K}
        )
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the RAG chain."""
        def format_docs(docs: List[Any]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "response_language": lambda x: x["response_language"]
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    async def get_answer(self, question: str, response_language: str = "English") -> str:
        """Get an answer using the RAG chain."""
        try:
            answer = await self.rag_chain.ainvoke({
                "question": question,
                "response_language": response_language
            })
            return answer
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            raise e
    
    def update_vectorstore(self, vectorstore: Chroma):
        """Update the vectorstore and reinitialize the chain."""
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type=config.RETRIEVAL_SEARCH_TYPE,
            search_kwargs={"k": config.RETRIEVAL_K}
        )
        self._initialize_chain()