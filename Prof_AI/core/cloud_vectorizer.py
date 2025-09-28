"""
Cloud Vectorizer - Handles connection to ChromaDB Cloud
"""

import logging
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import config

class CloudVectorizer:
    """Handles connection and operations with ChromaDB Cloud."""

    def __init__(self):
        """Initializes the connection to ChromaDB Cloud."""
        if not all([config.CHROMA_CLOUD_API_KEY, config.CHROMA_CLOUD_TENANT, config.CHROMA_CLOUD_DATABASE]):
            raise ValueError("ChromaDB Cloud credentials are not fully configured in the .env file.")
        
        self.client = chromadb.CloudClient(
            api_key=config.CHROMA_CLOUD_API_KEY,
            tenant=config.CHROMA_CLOUD_TENANT,
            database=config.CHROMA_CLOUD_DATABASE
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME, 
            openai_api_key=config.OPENAI_API_KEY
        )
        logging.info("ChromaDB Cloud client initialized.")

    def get_vector_store(self):
        """
        Retrieves the existing vector store from ChromaDB Cloud.
        This is used for querying (e.g., in ChatService).
        """
        try:
            logging.info(f"Loading existing ChromaDB Cloud collection: {config.CHROMA_COLLECTION_NAME}")
            vector_store = Chroma(
                client=self.client,
                collection_name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
            return vector_store
        except Exception as e:
            logging.error(f"Failed to load ChromaDB collection: {e}")
            # This might happen if the collection doesn't exist yet.
            # It will be created when documents are added.
            return None

    def create_vector_store_from_documents(self, documents):
        """
        Creates a new vector store in ChromaDB Cloud from a list of documents.
        This is used for ingestion (e.g., in DocumentService).
        """
        if not documents:
            logging.error("Cannot create vector store: No documents provided.")
            return None
        
        try:
            logging.info(f"Creating/updating ChromaDB Cloud collection '{config.CHROMA_COLLECTION_NAME}' with {len(documents)} documents.")
            vector_store = Chroma.from_documents(
                client=self.client,
                documents=documents,
                embedding=self.embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
            logging.info("ChromaDB Cloud collection created/updated successfully.")
            return vector_store
        except Exception as e:
            logging.error(f"Failed to create ChromaDB collection from documents: {e}")
            raise