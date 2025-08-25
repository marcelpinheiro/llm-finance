import os
from typing import List, Union
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class EmbeddingProvider:
    """Provider for generating embeddings with switchable backends."""
    
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the selected embedding provider."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI embedding provider."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        
        self.client = OpenAI(api_key=api_key)
        self.dimension = 1536  # text-embedding-3-small dimension
    
    def _init_huggingface(self):
        """Initialize Hugging Face embedding provider."""
        model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-small-en")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # BAAI/bge-small-en dimension
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        if self.provider == "openai":
            return self._openai_embedding(text)
        elif self.provider == "huggingface":
            return self._huggingface_embedding(text)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if self.provider == "openai":
            return self._openai_embeddings(texts)
        elif self.provider == "huggingface":
            return self._huggingface_embeddings(texts)
    
    def _openai_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text."""
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate OpenAI embeddings for multiple texts."""
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    
    def _huggingface_embedding(self, text: str) -> List[float]:
        """Generate Hugging Face embedding for text."""
        text = text.replace("\n", " ")
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def _huggingface_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate Hugging Face embeddings for multiple texts."""
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self.dimension

# Convenience functions
def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using configured provider."""
    provider = EmbeddingProvider()
    return provider.get_embedding(text)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts using configured provider."""
    provider = EmbeddingProvider()
    return provider.get_embeddings(texts)

def get_embedding_dimension() -> int:
    """Get the dimension of embeddings from configured provider."""
    provider = EmbeddingProvider()
    return provider.get_dimension()

if __name__ == "__main__":
    # Example usage
    provider = EmbeddingProvider()
    print(f"Using provider: {provider.provider}")
    print(f"Embedding dimension: {provider.get_dimension()}")
    
    # Test embedding generation
    test_text = "This is a test sentence for embedding."
    embedding = provider.get_embedding(test_text)
    print(f"Embedding length: {len(embedding)}")
