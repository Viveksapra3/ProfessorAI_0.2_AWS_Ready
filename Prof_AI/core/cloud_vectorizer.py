import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import config

class CloudVectorizer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.client = self._get_chroma_client()
    
    def _get_chroma_client(self):
        return chromadb.CloudClient(
            api_key=config.CHROMA_CLOUD_API_KEY,
            tenant=config.CHROMA_CLOUD_TENANT,
            database=config.CHROMA_CLOUD_DATABASE
        )
    
    def get_vector_store(self, collection_name=None):
        collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
