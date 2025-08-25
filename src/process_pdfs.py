import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from document_parser import DocumentParser
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def get_company_id(symbol):
    """Get company ID from database by symbol."""
    # Try to get DB_HOST from environment, with fallback logic
    db_host = os.getenv("DB_HOST", "localhost")
    
    # If running in Docker, the default should be "db" service name
    # Check if we're likely in a Docker container
    if os.path.exists("/.dockerenv") or "DBT_HOST" in os.environ:
        db_host = os.getenv("DBT_HOST", "db")
    
    conn = psycopg2.connect(
        host=db_host,
        database=os.getenv("DB_NAME", "llm_finance"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        port=os.getenv("DB_PORT", "5432")
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM finance.companies WHERE symbol = %s", (symbol,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return result[0] if result else None

def process_all_pdfs():
    """Process all PDFs in the data/sec_filings directory."""
    pdf_directory = "data/sec_filings"
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist.")
        return
    
    # Initialize document parser
    parser = DocumentParser()
    
    # Process each PDF
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            
            # Extract company symbol from filename (assuming format SYMBOL_10K_YEAR.pdf)
            symbol = filename.split("_")[0]
            
            # Get company ID from database
            company_id = get_company_id(symbol)
            if not company_id:
                print(f"Company {symbol} not found in database. Please ingest company data first.")
                continue
            
            # Extract year from filename
            try:
                year = filename.split("_")[2].split(".")[0]
                # Handle negative years (likely a typo in filename)
                if year.startswith("-"):
                    year = year[1:]  # Remove the negative sign
            except:
                year = "Unknown"
            
            print(f"Processing {filename} for company {symbol} (ID: {company_id})")
            
            try:
                # Process the PDF
                documents = parser.parse_pdf_from_file(
                    file_path=file_path,
                    company_id=company_id,
                    document_type="10k",
                    filing_date=f"{year}-12-31" if year != "Unknown" else None,  # Assuming fiscal year end
                    title=f"{symbol} 10-K {year}"
                )
                
                print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def process_single_pdf(file_path, symbol, year):
    """Process a single PDF file."""
    # Get company ID from database
    company_id = get_company_id(symbol)
    if not company_id:
        print(f"Company {symbol} not found in database. Please ingest company data first.")
        return
    
    # Initialize document parser
    parser = DocumentParser()
    
    print(f"Processing {file_path} for company {symbol} (ID: {company_id})")
    
    try:
        # Process the PDF
        documents = parser.parse_pdf_from_file(
            file_path=file_path,
            company_id=company_id,
            document_type="10k",
            filing_date=f"{year}-12-31",  # Assuming fiscal year end
            title=f"{symbol} 10-K {year}"
        )
        
        print(f"Successfully processed {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process specific file if provided
        file_path = sys.argv[1]
        symbol = sys.argv[2] if len(sys.argv) > 2 else "UNKNOWN"
        year = sys.argv[3] if len(sys.argv) > 3 else "Unknown"
        
        if os.path.exists(file_path):
            process_single_pdf(file_path, symbol, year)
        else:
            print(f"File {file_path} does not exist.")
    else:
        # Process all PDFs
        process_all_pdfs()
