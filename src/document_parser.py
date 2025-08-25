import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from llama_parse import LlamaParse
from PyPDF2 import PdfReader
import psycopg2
import tempfile
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

class DocumentParser:
    """Parse and process financial documents like annual reports and 10-K filings."""
    
    def __init__(self):
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
            language="en",
            parsing_instruction="Extract all text content from this financial document, including tables, figures, and detailed sections."
        )
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
    
    def parse_pdf_from_file(self, file_path: str, company_id: int, document_type: str, filing_date: str = None, title: str = None) -> List[Dict]:
        """
        Parse a PDF document from a file path and extract text content.
        
        Args:
            file_path (str): Path to the PDF file
            company_id (int): ID of the company in the database
            document_type (str): Type of document (e.g., 'annual_report', '10k')
            filing_date (str): Date of filing (optional)
            title (str): Title of the document (optional)
            
        Returns:
            List[Dict]: List of parsed document sections
        """
        try:
            # Parse document with LlamaParse
            documents = self.parser.load_data(file_path)
            
            # Check if we got enough content, if not, try PyPDF2 as fallback
            content = documents[0].text if documents else ""
            if len(content) < 5000:  # If less than 5000 characters, try fallback
                print(f"LlamaParse extracted only {len(content)} characters, trying PyPDF2 fallback...")
                fallback_content = self._extract_with_pypdf2(file_path)
                if len(fallback_content) > len(content):
                    print(f"PyPDF2 extracted {len(fallback_content)} characters, using fallback content")
                    content = fallback_content
                else:
                    print("PyPDF2 didn't extract more content, using LlamaParse result")
            
            # Save document to database
            self._save_document_to_db(company_id, document_type, filing_date, title, content)
            
            return [{"text": content, "metadata": documents[0].metadata if documents else {}}]
            
        except Exception as e:
            print(f"Error parsing PDF from file {file_path}: {e}")
            raise
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """
        Extract text from PDF using PyPDF2 as fallback.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def _save_document_to_db(self, company_id: int, document_type: str, filing_date: str, title: str, content: str):
        """
        Save parsed document content to the database.
        
        Args:
            company_id (int): ID of the company in the database
            document_type (str): Type of document
            filing_date (str): Date of filing
            title (str): Title of the document
            content (str): Document content
        """
        cursor = self.db_connection.cursor()
        
        try:
            # Insert document
            cursor.execute("""
                INSERT INTO finance.documents
                (company_id, document_type, filing_date, title, content)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (company_id, document_type, filing_date if filing_date else None, title, content))
            
            document_id = cursor.fetchone()[0]
            self.db_connection.commit()
            
            print(f"Successfully saved document to database with ID {document_id}")
            return document_id
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error saving document to database: {e}")
            raise
        finally:
            cursor.close()
    
    def close_connection(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()

# Example usage
if __name__ == "__main__":
    parser = DocumentParser()
    
    try:
        # Example: Parse a PDF from file
        # documents = parser.parse_pdf_from_file(
        #     file_path="./data/annual_reports/example_report.pdf",
        #     company_id=1,
        #     document_type="annual_report",
        #     title="2023 Annual Report"
        # )
        
        pass
        
    finally:
        parser.close_connection()
