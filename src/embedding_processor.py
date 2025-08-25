import psycopg2
import os
from typing import List, Tuple
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from embeddings import EmbeddingProvider

# Load environment variables
load_dotenv()

class EmbeddingProcessor:
    """Process document chunks and generate embeddings for storage in the database."""
    
    def __init__(self):
        self.embedding_provider = EmbeddingProvider()
        self.db_connection = None
        self._connect_to_db()
    
    def _connect_to_db(self):
        """Establish database connection."""
        # Try to get DB_HOST from environment, with fallback logic
        db_host = os.getenv("DB_HOST", "localhost")
        
        # If running in Docker, the default should be "db" service name
        # Check if we're likely in a Docker container
        if os.path.exists("/.dockerenv") or "DBT_HOST" in os.environ:
            db_host = os.getenv("DBT_HOST", "db")
        
        try:
            self.db_connection = psycopg2.connect(
                host=db_host,
                database=os.getenv("DB_NAME", "llm_finance"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres"),
                port=os.getenv("DB_PORT", "5432")
            )
            print(f"Successfully connected to database at {db_host}")
        except Exception as e:
            print(f"Error connecting to database at {db_host}: {e}")
            raise
    
    def process_document_embeddings(self, document_id: int):
        """
        Process a document by chunking its content and generating embeddings.
        
        Args:
            document_id (int): ID of the document in the database
        """
        cursor = self.db_connection.cursor()
        
        try:
            # Get document content
            cursor.execute("""
                SELECT content FROM finance.documents WHERE id = %s
            """, (document_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"Document with ID {document_id} not found")
                return
            
            content = result[0]
            
            # Chunk document
            chunks = self._chunk_text(content)
            print(f"Document chunked into {len(chunks)} chunks")
            
            # Generate embeddings for chunks
            embeddings = self.embedding_provider.get_embeddings(chunks)
            print(f"Generated embeddings for {len(embeddings)} chunks")
            
            # Save embeddings to database
            self._save_embeddings_to_db(document_id, chunks, embeddings)
            
            print(f"Successfully processed embeddings for document ID {document_id}")
            
        except Exception as e:
            print(f"Error processing document embeddings for document ID {document_id}: {e}")
            raise
        finally:
            cursor.close()
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk in characters
            overlap (int): Overlap between chunks in characters
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # Move start position with overlap
            
            # If we've reached the end of the text, break
            if end >= len(text):
                break
        
        return chunks
    
    def _save_embeddings_to_db(self, document_id: int, chunks: List[str], embeddings: List[List[float]]):
        """
        Save embeddings to the database.
        
        Args:
            document_id (int): ID of the document
            chunks (List[str]): Text chunks
            embeddings (List[List[float]]): Embeddings for each chunk
        """
        cursor = self.db_connection.cursor()
        
        try:
            # Insert embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Convert embedding to PostgreSQL vector format
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                cursor.execute("""
                    INSERT INTO finance.document_embeddings
                    (document_id, chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (document_id, i, chunk, embedding_str))
            
            self.db_connection.commit()
            print(f"Successfully saved {len(embeddings)} embeddings to database")
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error saving embeddings to database: {e}")
            raise
        finally:
            cursor.close()
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar document chunks using vector similarity.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk_text, similarity_score) tuples
        """
        cursor = self.db_connection.cursor()
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_provider.get_embedding(query)
            
            # Search for similar chunks
            cursor.execute("""
                SELECT chunk_text, 1 - (embedding <=> %s) as similarity
                FROM finance.document_embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]
            
        except Exception as e:
            print(f"Error searching similar chunks: {e}")
            raise
        finally:
            cursor.close()
    
    def close_connection(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()
    
    def process_all_documents(self):
        """Process embeddings for all documents in the database."""
        cursor = self.db_connection.cursor()
        
        try:
            # Get all document IDs
            cursor.execute("SELECT id FROM finance.documents")
            document_ids = [row[0] for row in cursor.fetchall()]
            
            print(f"Found {len(document_ids)} documents to process")
            
            # Process each document
            for document_id in document_ids:
                print(f"Processing document ID {document_id}")
                self.process_document_embeddings(document_id)
                
        except Exception as e:
            print(f"Error processing documents: {e}")
            raise
        finally:
            cursor.close()

# Example usage
if __name__ == "__main__":
    processor = EmbeddingProcessor()
    
    try:
        # Process embeddings for all documents
        processor.process_all_documents()
        
    finally:
        processor.close_connection()
