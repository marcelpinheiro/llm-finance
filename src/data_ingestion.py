import yfinance as yf
import pandas as pd
import psycopg2
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Top 10 companies by market cap (as of recent data)
TOP_COMPANIES = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "NVDA",  # NVIDIA
    "META",  # Meta/Facebook
    "TSLA",  # Tesla
    "BRK-B", # Berkshire Hathaway
    "LLY",   # Eli Lilly
    "V"      # Visa
]

class FinancialDataIngestor:
    """Ingest financial data from various sources into the database."""
    
    def __init__(self):
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
    
    def ingest_companies(self, symbols: List[str] = None):
        """
        Ingest company information into the database.
        
        Args:
            symbols (List[str]): List of stock symbols to ingest. If None, uses TOP_COMPANIES.
        """
        if symbols is None:
            symbols = TOP_COMPANIES
        
        cursor = self.db_connection.cursor()
        
        for symbol in symbols:
            try:
                # Get company info from yfinance
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract relevant information
                name = info.get('longName', info.get('shortName', symbol))
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                market_cap = info.get('marketCap', None)
                
                # Insert or update company information
                cursor.execute("""
                    INSERT INTO finance.companies (symbol, name, sector, industry, market_cap)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) 
                    DO UPDATE SET 
                        name = EXCLUDED.name,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        market_cap = EXCLUDED.market_cap
                """, (symbol, name, sector, industry, market_cap))
                
                print(f"Successfully ingested company data for {symbol}")
                
            except Exception as e:
                print(f"Error ingesting data for {symbol}: {e}")
        
        self.db_connection.commit()
        cursor.close()
    
    def ingest_financial_statements(self, symbols: List[str] = None):
        """
        Ingest financial statements into the database.
        
        Args:
            symbols (List[str]): List of stock symbols to ingest. If None, uses TOP_COMPANIES.
        """
        if symbols is None:
            symbols = TOP_COMPANIES
        
        cursor = self.db_connection.cursor()
        
        for symbol in symbols:
            try:
                # Get company info from yfinance
                ticker = yf.Ticker(symbol)
                
                # Get company ID from database
                cursor.execute("SELECT id FROM finance.companies WHERE symbol = %s", (symbol,))
                result = cursor.fetchone()
                if not result:
                    print(f"Company {symbol} not found in database, skipping...")
                    continue
                
                company_id = result[0]
                
                # Get financial statements
                income_stmt = ticker.financials
                balance_sheet = ticker.balance_sheet
                cash_flow = ticker.cashflow
                
                # Process income statement
                if not income_stmt.empty:
                    for year in income_stmt.columns:
                        try:
                            fiscal_year = year.year
                            data = {
                                'revenue': income_stmt.loc['Total Revenue', year] if 'Total Revenue' in income_stmt.index else None,
                                'net_income': income_stmt.loc['Net Income', year] if 'Net Income' in income_stmt.index else None,
                            }
                            
                            # Handle NaN values by converting them to None
                            for key, value in data.items():
                                if pd.isna(value):
                                    data[key] = None
                            
                            # Insert financial statement
                            cursor.execute("""
                                INSERT INTO finance.financial_statements
                                (company_id, statement_type, fiscal_year, fiscal_period, data)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (company_id, statement_type, fiscal_year, fiscal_period)
                                DO UPDATE SET data = EXCLUDED.data
                            """, (company_id, 'income_statement', fiscal_year, 'Annual', json.dumps(data)))
                        except Exception as e:
                            print(f"Error processing income statement for {symbol} in {fiscal_year}: {e}")
                            # Continue with other years
                
                # Process balance sheet
                if not balance_sheet.empty:
                    for year in balance_sheet.columns:
                        try:
                            fiscal_year = year.year
                            data = {
                                'total_assets': balance_sheet.loc['Total Assets', year] if 'Total Assets' in balance_sheet.index else None,
                                'total_liabilities': balance_sheet.loc['Total Liabilities', year] if 'Total Liabilities' in balance_sheet.index else None,
                                'shareholders_equity': balance_sheet.loc['Total Stockholder Equity', year] if 'Total Stockholder Equity' in balance_sheet.index else None,
                            }
                            
                            # Handle NaN values by converting them to None
                            for key, value in data.items():
                                if pd.isna(value):
                                    data[key] = None
                            
                            # Insert financial statement
                            cursor.execute("""
                                INSERT INTO finance.financial_statements
                                (company_id, statement_type, fiscal_year, fiscal_period, data)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (company_id, statement_type, fiscal_year, fiscal_period)
                                DO UPDATE SET data = EXCLUDED.data
                            """, (company_id, 'balance_sheet', fiscal_year, 'Annual', json.dumps(data)))
                        except Exception as e:
                            print(f"Error processing balance sheet for {symbol} in {fiscal_year}: {e}")
                            # Continue with other years
                
                # Process cash flow
                if not cash_flow.empty:
                    for year in cash_flow.columns:
                        try:
                            fiscal_year = year.year
                            data = {
                                'operating_cash_flow': cash_flow.loc['Operating Cash Flow', year] if 'Operating Cash Flow' in cash_flow.index else None,
                                'investing_cash_flow': cash_flow.loc['Investing Cash Flow', year] if 'Investing Cash Flow' in cash_flow.index else None,
                                'financing_cash_flow': cash_flow.loc['Financing Cash Flow', year] if 'Financing Cash Flow' in cash_flow.index else None,
                            }
                            
                            # Handle NaN values by converting them to None
                            for key, value in data.items():
                                if pd.isna(value):
                                    data[key] = None
                            
                            # Insert financial statement
                            cursor.execute("""
                                INSERT INTO finance.financial_statements
                                (company_id, statement_type, fiscal_year, fiscal_period, data)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (company_id, statement_type, fiscal_year, fiscal_period)
                                DO UPDATE SET data = EXCLUDED.data
                            """, (company_id, 'cash_flow', fiscal_year, 'Annual', json.dumps(data)))
                        except Exception as e:
                            print(f"Error processing cash flow for {symbol} in {fiscal_year}: {e}")
                            # Continue with other years
                
                print(f"Successfully ingested financial statements for {symbol}")
                
            except Exception as e:
                print(f"Error ingesting financial statements for {symbol}: {e}")
                # Continue with other symbols instead of aborting transaction
        
        self.db_connection.commit()
        cursor.close()
    
    def ingest_stock_prices(self, symbols: List[str] = None, period: str = "5y"):
        """
        Ingest historical stock prices into the database.
        
        Args:
            symbols (List[str]): List of stock symbols to ingest. If None, uses TOP_COMPANIES.
            period (str): Period for historical data (e.g., "1y", "5y", "max")
        """
        if symbols is None:
            symbols = TOP_COMPANIES
        
        cursor = self.db_connection.cursor()
        
        for symbol in symbols:
            try:
                # Get company ID from database
                cursor.execute("SELECT id FROM finance.companies WHERE symbol = %s", (symbol,))
                result = cursor.fetchone()
                if not result:
                    print(f"Company {symbol} not found in database, skipping...")
                    continue
                
                company_id = result[0]
                
                # Get historical data from yfinance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                # Insert stock prices
                for date, row in hist.iterrows():
                    try:
                        cursor.execute("""
                            INSERT INTO finance.stock_prices
                            (company_id, date, open_price, close_price, high_price, low_price, volume, adjusted_close)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (company_id, date)
                            DO UPDATE SET
                                open_price = EXCLUDED.open_price,
                                close_price = EXCLUDED.close_price,
                                high_price = EXCLUDED.high_price,
                                low_price = EXCLUDED.low_price,
                                volume = EXCLUDED.volume,
                                adjusted_close = EXCLUDED.adjusted_close
                        """, (
                            company_id,
                            date.date(),
                            float(row['Open']) if not pd.isna(row['Open']) else None,
                            float(row['Close']) if not pd.isna(row['Close']) else None,
                            float(row['High']) if not pd.isna(row['High']) else None,
                            float(row['Low']) if not pd.isna(row['Low']) else None,
                            int(row['Volume']) if not pd.isna(row['Volume']) else None,
                            float(row['Close']) if not pd.isna(row['Close']) else None  # Using close as adjusted_close for simplicity
                        ))
                    except Exception as e:
                        print(f"Error processing stock price for {symbol} on {date.date()}: {e}")
                        # Continue with other dates
                
                print(f"Successfully ingested stock prices for {symbol}")
                
            except Exception as e:
                print(f"Error ingesting stock prices for {symbol}: {e}")
                # Continue with other symbols instead of aborting transaction
        
        self.db_connection.commit()
        cursor.close()
    
    def close_connection(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()

# Example usage
if __name__ == "__main__":
    ingestor = FinancialDataIngestor()
    
    try:
        # Ingest company information
        ingestor.ingest_companies()
        
        # Ingest financial statements
        ingestor.ingest_financial_statements()
        
        # Ingest stock prices
        ingestor.ingest_stock_prices()
        
    finally:
        ingestor.close_connection()
